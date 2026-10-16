#!/usr/bin/env python3
"""B70: round the token count of eager (non-graph) steps up to a bucket, so oneDNN int4 GEMM primitives get reused.

vllm-xpu-kernels' int4_gemm_w4a16 (the only XPU dense int4 path) caches one oneDNN primitive per exact
(M, N, K, ...) in a 512-entry LRU shared by every int4 layer (onednn_ext.h:800-880). A step whose token count M was
not seen before builds a new primitive for every int4 weight shape: 20-50 ms each on the B70 (hunt B19-1 microbench:
first call 20-53 ms vs 0.1-1 ms warm), i.e. ~0.1-0.3 s for the ~6 int4 shapes of Qwen3.8-27B, inside the forward step.
Every step larger than the biggest graph capture size runs eager with M = prefill chunk + decode tokens, so a new
prompt length (its final chunk) or a new mix of prefill + decoding streams almost always hits a cold M: TTFT grows and
every concurrent stream stalls for that step. Past ~512/6 distinct M the LRU churns and it never warms up.

Padding the step to a multiple of B70_EAGER_PAD_TOKENS (default 128) bounds the distinct M to 8192/128 = 64 per
shape (384 entries for 6 shapes, under the LRU size). The V2 runner already pads graph batches this way: the extra
rows get PAD_SLOT_ID slot mappings (no KV writes) and attention/GDN metadata use the real token count; logits only
read real rows. Cost: <= 127 padded rows per eager step (<2% of an 8K-token prefill step, more on short prompts).
The rounded count never exceeds max_num_batched_tokens (the input buffer size). Torch's own oneDNN matmuls for the
bf16/fp16 linears get the same bucketing. B70_EAGER_PAD_TOKENS=0 switches it off.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MARKER = "B70_EAGER_PAD_TOKENS"

IMPORT_OLD = "from dataclasses import replace\n"
IMPORT_NEW = '''from dataclasses import replace

# B70_EAGER_PAD_TOKENS: bucket for the token count of eager steps (0/1 = off)
_B70_EAGER_PAD = int(__import__("os").environ.get("B70_EAGER_PAD_TOKENS", "128") or 0)
'''

PAD_ANCHOR = '''        if batch_desc.num_tokens == 0:
            # All DP ranks have zero tokens to run.
'''
PAD_NEW = '''        # B70_EAGER_PAD_TOKENS: bucket eager token counts so the per-M oneDNN
        # primitives (int4 GEMM cache keyed on exact M) are reused.
        if (
            _B70_EAGER_PAD > 1
            and not dummy_run
            and batch_desc.cg_mode == CUDAGraphMode.NONE
            and batch_desc.num_tokens > 0
            and self.pcp_manager is None
            and self.dp_size == 1
        ):
            _b70_padded = min(
                -(-batch_desc.num_tokens // _B70_EAGER_PAD) * _B70_EAGER_PAD,
                self.max_num_tokens,
            )
            if _b70_padded > batch_desc.num_tokens:
                batch_desc = replace(batch_desc, num_tokens=_b70_padded)
''' + PAD_ANCHOR


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    for old, what in ((IMPORT_OLD, "dataclasses import"), (PAD_ANCHOR, "zero-token batch check")):
        if text.count(old) != 1:
            raise RuntimeError(f"V2 model_runner {what} anchor changed or missing; refusing")
    for name in ("CUDAGraphMode", "self.pcp_manager", "self.dp_size", "self.max_num_tokens"):
        if name not in text:
            raise RuntimeError(f"V2 model_runner lacks {name}; refusing")
    return text.replace(IMPORT_OLD, IMPORT_NEW, 1).replace(PAD_ANCHOR, PAD_NEW, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = Path(next(iter(spec.submodule_search_locations))) / "v1/worker/gpu/model_runner.py"
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
