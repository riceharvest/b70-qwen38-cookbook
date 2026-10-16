#!/usr/bin/env python3
"""B70: pick the decode split-K count from batch and KV heads while an XPU graph is captured (q=1 decode).

Under FULL XPU-graph capture the attention metadata carries max_seqlen_k = max_model_len (262144 here), and
vllm-xpu-kernels derives num_kv_splits from it on the host. The captured graph then replays that long-context split
count at every real context length. One decode work-group per Xe core (num_splits = xe_cores // (batch * kv_heads))
was the best context-independent count in a B70 sweep (hunt B09: ctx 2K 1.25-1.91x, 8K 1.13-1.41x, 32K 1.05-1.15x
faster per attention call, batch 1-8; outputs within one bf16 ulp of eager). Larger batches, sliding-window layers
and eager calls keep the kernel heuristic.

On this setup the q=1 calls are the MTP draft steps and non-speculative decode; the q=k+1 verify calls go through
runtime/spec_packed.py, which applies the same rule to its packed rows.
Backport of riceharvest/vllm hunt-xpu-graph-kv-splits (3152405 + dd74ab6). Switch off with B70_GRAPH_KV_SPLITS=0.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MARKER = "B70_GRAPH_KV_SPLITS"

HELPER_ANCHOR = "\n_OPS_REGISTERED = False\n"
HELPER = '''
# B70_GRAPH_KV_SPLITS: decode split-K count under XPU-graph capture (see cookbook patch_xpu_graph_kv_splits.py)
_B70_XE_CORES: dict[int, int] = {}


def _b70_num_xe_cores(device: torch.device) -> int:
    # torch exposes the Xe-core count as gpu_subslice_count (32 on Arc Pro B70); vllm_xpu_kernels'
    # _infer_num_xe_cores looks for property names torch 2.14 does not expose and falls back to 20.
    idx = device.index if device.index is not None else torch.xpu.current_device()
    n = _B70_XE_CORES.get(idx)
    if n is None:
        try:
            n = int(getattr(torch.xpu.get_device_properties(idx), "gpu_subslice_count", 0) or 0)
        except Exception:
            n = 0
        _B70_XE_CORES[idx] = n
    return n


def _b70_graph_decode_num_kv_splits(q, k, cu_seqlens_q, max_seqlen_q, block_table, seqused_k, window_size):
    import os

    if os.environ.get("B70_GRAPH_KV_SPLITS", "1") == "0":
        return None
    if max_seqlen_q != 1 or block_table is None or seqused_k is None:
        return None
    if window_size is not None and tuple(window_size) != (-1, -1):
        return None  # xk already sizes splits from the window
    is_capturing = getattr(torch.xpu, "is_current_stream_capturing", None)
    if is_capturing is None or not is_capturing():
        return None
    cores = _b70_num_xe_cores(q.device)
    wgs_per_split = (cu_seqlens_q.numel() - 1) * k.size(2)
    if cores <= 0 or wgs_per_split <= 0 or wgs_per_split > cores:
        return None
    return min(32, cores // wgs_per_split)

'''

CALL_OLD = '''        return flash_attn_varlen_func(
            out=out,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
            s_aux=s_aux,
            window_size=real_window_size,
            # alibi_slopes = alibi_slopes,
            # softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
'''

CALL_NEW = '''        # B70_GRAPH_KV_SPLITS
        extra_kwargs = {}
        num_splits_kv = _b70_graph_decode_num_kv_splits(
            q, k, cu_seqlens_q, max_seqlen_q, block_table, seqused_k, real_window_size
        )
        if num_splits_kv is not None:
            extra_kwargs["num_splits_kv"] = num_splits_kv
        return flash_attn_varlen_func(
            out=out,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
            s_aux=s_aux,
            window_size=real_window_size,
            # alibi_slopes = alibi_slopes,
            # softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            **extra_kwargs,
        )
'''


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    if text.count(HELPER_ANCHOR) != 1:
        raise RuntimeError("_xpu_ops.py _OPS_REGISTERED anchor changed; refusing")
    if text.count(CALL_OLD) != 1:
        raise RuntimeError("xpu_ops.flash_attn_varlen_func call anchor changed; refusing")
    text = text.replace(HELPER_ANCHOR, "\n" + HELPER + HELPER_ANCHOR.lstrip("\n"), 1)
    return text.replace(CALL_OLD, CALL_NEW, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = Path(next(iter(spec.submodule_search_locations))) / "_xpu_ops.py"
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    original = path.read_text()
    patched = patch_text(original)
    if patched == original:
        print(f"already patched {path}")
        return
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
