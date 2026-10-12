# Attention kernels

`serve.sh` mounts `libattn_kernels_xe_2.b70.so` over the attention library in the vLLM XPU image. Download it from the
[GitHub release](../../releases) into this folder, or rebuild it with `build.sh`. Without it, `serve.sh` falls back to the
stock kernels. That still works, just slower.

It's built from vllm-xpu-kernels `release/0.1.15.4` + sycl-tla `87f68506` (what the pinned `vllm/vllm-openai-xpu:v0.31.0`
image ships) plus two patches. **It only fits that image.** If you change the image, drop the mount or rebuild.

| Patch | What it does | Effect on a B70 |
|---|---|---|
| `tla-fast3-b70.patch` (sycl-tla) | fp8 e4m3 -> fp16 conversion in 3 instructions per 64 values instead of 11: sign-extending shift, no NaN handling, and the x256 exponent bias left in | prefill attention with fp8 KV 41 -> 72 TFLOPS (same as fp16 KV) |
| `xk-fold-b70.patch` (vllm-xpu-kernels) | folds that x256 into the softmax and V scales; adds a per-row causal mask, so MTP verify reads the KV cache once (all 5 positions x GQA heads packed into one decode pass); fixes the split-K decode epilogue for tiles with more than one subgroup along Q; uses _8 tiles for large GQA ratios at head 256 (the _16 tile spills) | 65K verify 671 -> 565 us/layer; decode q_len=1 a bit faster |
| `b70_prod_*.conf` | only the kernel variants this model uses (head 256 paged, vision head 72 -> 96 policy, 128) | smaller, faster build |

Outputs are bit-identical to stock kernels across all 254 non-NaN e4m3 codes (`bench/fmha_stress.py`, `bench/bitid.py`).
The q_len x GQA packed verify is enabled at runtime by `runtime/spec_packed.py` (`B70_SPEC_PACKED=1`). It falls back
to stock for sinks, sliding window, softcap, or q_len < 2.

There's also an opt-in 32-row verify tile (`B70_SPEC_Q32=1`). It's correct and 12-14% faster in isolation, but
slower inside the real server, so it's off.

Upstream versions of the fp8 conversion (opt-in, default behaviour unchanged):
[intel/sycl-tla#859](https://github.com/intel/sycl-tla/pull/859) +
[vllm-project/vllm-xpu-kernels#638](https://github.com/vllm-project/vllm-xpu-kernels/pull/638).

## Building

`ONEAPI=/path/to/oneapi ./build.sh` takes ~25 min on a 32-thread desktop (tested; the output is bit-identical to the release `.so`). It needs icpx 2026.0 and builds for BMG-G31 only
(`VLLM_XPU_XE2_AOT_DEVICES=bmg-g31-a0`). Most of the time goes into the serial GPU codegen at link time, so extra `-j`
barely helps. Each icpx job can take ~8 GB of RAM, so keep `JOBS` at 2-4.

## Licenses

vllm-xpu-kernels: Apache-2.0. sycl-tla: BSD-3-Clause (NVIDIA + Intel). The patches and the `.so` are derivative works
under those licenses. See the upstream repositories for the full texts.
