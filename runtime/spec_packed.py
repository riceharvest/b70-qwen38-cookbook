"""B70: single-pass speculative-verify attention for vllm_xpu_kernels (replaces _spec_decode_varlen_fwd).
All q_len draft positions x GQA heads of a sequence become packed rows of ONE decode query per KV head
(KV read once). Per-row causal limits are applied inside the decode kernel (B70_SPEC_QLEN, patched
paged_decode). Requires libattn_kernels_xe_2 built with the B70 spec mask."""
import os, torch
import vllm_xpu_kernels.flash_attn_interface as fai
_orig = fai._spec_decode_varlen_fwd
_DBG = os.environ.get('B70_SPEC_DEBUG') == '1'; _seen = set()

# B70_GRAPH_KV_SPLITS: under FULL XPU-graph capture max_seqlen_k is max_model_len, so the kernel's host heuristic picks
# its long-context split count (B=1/2/4/8 -> 32/17/8/4 here) and every replay keeps it at any real context length.
# Aim for one decode work-group per Xe core instead (hunt B09 sweep: best context-independent count for q=1 decode).
# Packed verify rows: the head_dim-256 decode kernel tiles a KV head's ql*G query rows into 8-row work-groups
# (16-row for head_dim <= 128 when the group exceeds 8), so a split costs B * Hkv * ceil(rows / tile) WGs.
_XE_CORES = {}

def _graph_splits(q, B, Hkv, rows, D):
    if os.environ.get("B70_GRAPH_KV_SPLITS", "1") == "0":
        return None
    cap = getattr(torch.xpu, "is_current_stream_capturing", None)
    if cap is None or not cap():
        return None
    dev = q.device.index if q.device.index is not None else torch.xpu.current_device()
    cores = _XE_CORES.get(dev)
    if cores is None:
        try:
            cores = int(getattr(torch.xpu.get_device_properties(dev), "gpu_subslice_count", 0) or 0)
        except Exception:
            cores = 0
        _XE_CORES[dev] = cores
    tile = 8 if (rows <= 8 or D > 128) else 16
    wgs = B * Hkv * ((rows + tile - 1) // tile)
    if cores <= 0 or wgs <= 0 or wgs > cores:
        return None
    return min(32, cores // wgs)

def _packed(q, k, v, out, cu_seqlens_q, seqused_k, block_table, max_seqlen_k, k_descale, v_descale,
            softmax_scale, s_aux, window_size, softcap):
    B = cu_seqlens_q.numel() - 1; ql = q.shape[0] // B; Hq, D = q.shape[1], q.shape[2]
    Hkv = k.shape[2]; G = Hq // Hkv
    if s_aux is not None or softcap != 0.0 or tuple(window_size) != (-1, -1) or ql < 2:
        return _orig(q, k, v, out, cu_seqlens_q, seqused_k, block_table, max_seqlen_k, k_descale, v_descale,
                     softmax_scale, s_aux, window_size, softcap)
    # [B*ql, Hkv*G, D] -> [B, Hkv*(ql*G), D] rows ordered (kv head, position, head-in-group)
    qp = q.view(B, ql, Hkv, G, D).permute(0, 2, 1, 3, 4).reshape(B, Hkv * ql * G, D)
    cu = torch.arange(B + 1, dtype=torch.int32, device=q.device)
    if _DBG and (B, ql, max_seqlen_k) not in _seen:
        _seen.add((B, ql, max_seqlen_k)); print(f"[B70 spec_packed] B={B} ql={ql} max_seqlen_k={max_seqlen_k} "
              f"capturing={torch.xpu.is_current_stream_capturing() if hasattr(torch.xpu,'is_current_stream_capturing') else '?'}", flush=True)
    os.environ["B70_SPEC_QLEN"] = str(ql)
    ns = _graph_splits(q, B, Hkv, ql * G, D)
    try:
        op = fai.flash_attn_varlen_func(qp, k, v, 1, cu, max_seqlen_k, seqused_k=seqused_k, causal=True,
                                        block_table=block_table, softmax_scale=softmax_scale,
                                        fa_version=2, k_descale=k_descale, v_descale=v_descale,
                                        num_splits_kv=ns)
    finally:
        os.environ["B70_SPEC_QLEN"] = "1"
    res = op.view(B, Hkv, ql, G, D).permute(0, 2, 1, 3, 4).reshape(B * ql, Hq, D)
    if out is not None:
        out.copy_(res); return out
    return res

if os.environ.get("B70_SPEC_PACKED", "1") == "1":
    fai._spec_decode_varlen_fwd = _packed
