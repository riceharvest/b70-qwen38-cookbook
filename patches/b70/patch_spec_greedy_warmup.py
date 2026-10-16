#!/usr/bin/env python3
"""B70: warm up the greedy rejection-sampling kernels at startup (backport of vLLM #59260, commit 2; fixes #54455).

The V2 kernel warmup runs every request with SamplingParams.for_sampler_warmup() (temperature 0.9), so the rejection
kernels only ever see fp32 processed logits. An all-greedy batch skips logits processing and hands them fp16 logits,
a different Triton specialization, so the first greedy request after every start compiles
_compute_local_logits_stats_kernel, _rejection_kernel and _resample_kernel in the request path (seen in the
orcavision-v3 log on 2026-10-15: "Triton kernel JIT compilation during inference"). A second warmup pass with
temperature=0 compiles them at startup instead. Startup only; serving is unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MARKER = "B70_SPEC_GREEDY_WARMUP"

CALL_OLD = '''    try:
        _warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens)
    finally:
'''
CALL_NEW = '''    try:
        _warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens)
        if rejection_sampler is not None:
            # B70_SPEC_GREEDY_WARMUP (vLLM #59260): greedy batches skip logits
            # processing, so the rejection kernels also see model-dtype logits.
            _warmup_kernels(
                model_runner, worker_execute_model, worker_sample_tokens, greedy=True
            )
    finally:
'''

SIG_OLD = '''    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    if model_runner.vllm_config.is_mm_encoder_only:
        return
'''
SIG_NEW = '''    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    greedy: bool = False,
) -> None:
    if model_runner.vllm_config.is_mm_encoder_only:
        return
'''

PARAMS_OLD = '''        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None
'''
PARAMS_NEW = '''        sampling_params = (
            SamplingParams(temperature=0)
            if greedy
            else SamplingParams.for_sampler_warmup()
        )
        pooling_params = None
'''


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    for old, what in ((CALL_OLD, "warmup_kernels call"), (SIG_OLD, "_warmup_kernels signature"),
                      (PARAMS_OLD, "warmup sampling params")):
        if text.count(old) != 1:
            raise RuntimeError(f"V2 warmup {what} anchor changed or missing; refusing")
    return text.replace(CALL_OLD, CALL_NEW, 1).replace(SIG_OLD, SIG_NEW, 1).replace(PARAMS_OLD, PARAMS_NEW, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = Path(next(iter(spec.submodule_search_locations))) / "v1/worker/gpu/warmup.py"
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
