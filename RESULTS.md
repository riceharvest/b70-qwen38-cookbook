# Results per release

What each release changed, measured on one Intel Arc Pro B70 (32 GB, stock 275 W), one request at a time, greedy,
512-token answers. Details and methods are in the [README](README.md). Newest first.

## image b70.4 (2026-10-15): bf16 copies of the fp8 linears freed by default

Image `ghcr.io/riceharvest/b70-vllm:v0.31.0-b70.4` (= `latest`), no new kernels (same .so files as v3). The fp8
weight-only linears no longer keep their bf16 copy: Qwen3.6-35B-A3B weights 19.13 instead of 21.75 GiB, same speed
(MTP4 kept vs freed, alternating in one session: 212.2 / 143.5 vs 213.0 / 142.7 and 206.9 / 135.0 vs 210.3 / 141.6).
On b70.2 freeing faulted the GPU under MTP; on b70.3 it never reproduced: eager and XPU-graph probes found nothing that
reads the freed weights, MTP4 under load with and without prefix caching, and a 10 min mixed stress (8 concurrent:
thinking QA, 3000-token code, ~10K-token contexts; 184 requests, 0 garbage, 0 xe faults). `B70_LINEAR_FP8_FREE=0` keeps
the copy. Image test: 0/32 corrupted outputs on MoE and on the dense 27B; dense speed identical to b70.3 in the same
session (101.4 / 72.6 vs 101.2 / 72.9).

## v3 (2026-10-15): Qwen3.6-35B-A3B MoE

[Release](https://github.com/riceharvest/b70-qwen38-cookbook/releases/tag/v3) · image `ghcr.io/riceharvest/b70-vllm:v0.31.0-b70.3`

New: int4 MoE decode GEMV (`kernels/moe-gemv/`), fp8 weight-only for the linears a checkpoint ships in bf16
(`patches/b70/patch_linear_fp8.py`). palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4, 16K context, decode tok/s code / prose:

| setup | no MTP | MTP k=4 |
|---|---|---|
| stock vLLM XPU v0.31 image | 75.6 / 76.3 | 123.4 / 113.2 (fits <= 25 GB only at 2K context) |
| v2 (cookbook patches + fp8 lm_head) | 78.7 / 79.1 | 183.9 / 134.3 |
| + int4 MoE decode GEMV | 93.8 / 94.3 | 200.8 / 145.4 |
| **v3: + fp8 attention / GDN / shared expert** | **118.9 / 119.6** | **243.7 / 164.4** |

vs stock: **+57% without MTP, +97% / +45% with MTP**. 64 concurrent users (no MTP, `--mamba-ssm-cache-dtype float16`): 1420-1510 tok/s generation, 2581-2769 total (two runs), gsm8k 98/100
([details](https://riceharvest.github.io/awesome-b70/results/vllm-64-users.html)). fp8 linears: KL vs bf16 weights 0.019, top-1 agreement 95.2%.
0/32 corrupted outputs with 8 concurrent thinking-length requests. Qwen3.8-27B unchanged from v2.

Proposed upstream from this release: int4 MoE GEMV as
[vllm-xpu-kernels #644](https://github.com/vllm-project/vllm-xpu-kernels/pull/644) and the fp8 lm_head as
[vllm #61815](https://github.com/vllm-project/vllm/pull/61815). On vLLM nightly they take decode from 75.8 / 76.3 to
89.4 / 90.2 tok/s, and from 157.3 / 118.7 to 183.7 / 131.9 with MTP.

## v2 (2026-10-15): packed MTP verify fixed

[Release](https://github.com/riceharvest/b70-qwen38-cookbook/releases/tag/v2) · image `ghcr.io/riceharvest/b70-vllm:v0.31.0-b70.2`

Fix: with the v1 kernel, packed MTP verify could output `!!!!` (a fully masked KV split gave a softmax sum of 0, and
0 x 1/0 = NaN). The v2 kernel adds a zero-sum guard, and packed verify is on by default again.

| check | v1 kernel | v2 kernel |
|---|---|---|
| kernel A/B, predicted NaN cases | 28 / 28 NaN | 0 |
| serving, 8 concurrent thinking-length answers | 2-3 / 16 corrupted per run | 0 / 48 |
| single stream at 131K context, packed on, code / prose | (off by default) | 82 / 77 tok/s (off: 80 / 77) |

## v1 (2026-10-12): Qwen3.8-27B on one B70

[Release](https://github.com/riceharvest/b70-qwen38-cookbook/releases/tag/v1)

Our asymmetric GPTQ int4 quant, fp8-KV attention kernels, packed MTP verify, fp8 lm_head, 262K context. Compared with
the setup we started from ([SergiioB's cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) on
the same card), same benchmark harness:

| | start | v1 | change |
|---|---|---|---|
| decode tok/s, code | 83 | 110 | +33% |
| decode tok/s, json | 93 | 128 | +38% |
| decode tok/s, prose | 52 | 64 | +23% |
| decode tok/s, reasoning | 75 | 105 | +40% |
| prefill tok/s, 6K / 24K / 48K / 97K prompt | 1906 / 1559 / 1273 / 865 | 2050 / 1834 / 1625 / ~1300 | +8% to +50% |
| context | 131K | 262K | 2x |
| KL vs BF16 (top-20 logits) | 0.069 | 0.039 | half |
| fp8-KV prefill attention kernel | 41 TFLOPS | 72 TFLOPS | +76% |

Upstream from this release: [intel/sycl-tla #859](https://github.com/intel/sycl-tla/pull/859),
[vllm-xpu-kernels #638](https://github.com/vllm-project/vllm-xpu-kernels/pull/638).
