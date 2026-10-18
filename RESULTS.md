# Results per release

What each release changed, measured on one Intel Arc Pro B70 (32 GB, stock 275 W), one request at a time, greedy,
512-token answers. Details and methods are in the [README](README.md). Newest first.

## image b70.7 (2026-10-18): rebased onto vLLM v0.32.0

Image `ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7`: same patches and kernels on the `vllm/vllm-openai-xpu:v0.32.0`
base. v0.32 ships vllm-xpu-kernels 0.1.15.4 and torch 2.14 like v0.31 (byte-identical kernel `.so` files), so the B70
attention kernel and the MoE GEMV carry over unchanged. Three patches are gone because v0.32 does the same upstream:
`patch_gdn_spec_sync`, `patch_gdn_zero_draft` (#56531 + #55504) and SergiioB's `patch_mtp_boundary`.
`patch_gdn_mixed_no_sync` now also covers the GDN state-index re-gather of the 2nd and 3rd GDN KV-cache groups
(`update_block_table`), which still synced 4 times per mixed prefill + MTP step on v0.31.

Same box, same config as the b70.5 image test (Qwen3.8-27B GPTQ-Int4-asym (Orca), MTP k=4, greedy, 512-token answers,
`GPU_UTIL=0.75`, 64K context), b70.6 numbers from the 2026-10-16 A/B:

| | code | json | prose | thinking | 32K context |
|---|---|---|---|---|---|
| b70.6 tok/s (tok/step) | 113.6 (3.76) | 124.8 (4.13) | 69.2 (2.29) | 107.9 (3.58) | 36.8-39.0 ms/step |
| b70.7 tok/s (tok/step) | 114.2 (3.76) | 127.4 (4.20) | 69.3 (2.29) | 110.0 (3.63) | 36.7 ms/step |

- KV pool at the same `GPU_UTIL`: 126,552 -> 135,591 tokens (+7%); weights 13.57 -> 13.48 GiB.
- 30-vs-14 canary: thinking off 5/6 (same as b70.5), thinking on 12/12; no `!!!!` output.
- Qwen3.6-35B-A3B (palmfuture GPTQ-Int4, `--language-model-only`, 16K, `GPU_UTIL=0.80`): MTP4 decode 264.2 / 177.9
  tok/s code / prose (b70.3 session: 243.7 / 164.4; different day, the box drifts 5-10%), prefill 4K / 15K 10.2K / 9.2K
  tok/s (unchanged), KV pool 76,878 tokens, 0/16 corrupted outputs in 8-concurrent thinking-length chats, thinking on
  12/12. Thinking off 4/6: both `sum(i*i for i in range(4))` phrasings answer 30, the known symmetric-int4 failure
  ([SergiioB](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)); not compared against b70.6 yet.

## image b70.6 (2026-10-16): faster long-context decode, correctness fixes, client compatibility

Image `ghcr.io/riceharvest/b70-vllm:v0.31.0-b70.6` (= `latest`), same kernels as b70.5. Interleaved A/B on the same
box, Qwen3.8-27B GPTQ-Int4-asym (Orca), MTP k=4, greedy, 512-token answers, b70.5-equivalent -> b70.6:

| ms per decode step | short prompt | 32K context | 64K context |
|---|---|---|---|
| b70.5 | 34.3 | 39.0 | 42.8 |
| b70.6 | 33.2 | 37.7 | 40.3 (74 -> 78.6 tok/s) |

Tokens per step and prefill unchanged. What changed:
- MTP verify and draft attention under XPU graphs used the split count of a 262K-token context at every length. The
  split count now follows a B70 sweep (`patch_xpu_graph_kv_splits.py`, `runtime/spec_packed.py`): verify attention at
  65K 544 -> 475 us/layer, draft decode 353 -> 275 us/layer.
- GDN steps that mix a prefill with MTP decodes no longer block on host-mask gathers (`patch_gdn_mixed_no_sync.py`).
- Decode rows with 0 or fewer than k drafts stay on the GDN speculative path (`patch_gdn_zero_draft.py`, backport of
  vLLM #56531 + #55504): before, such a row continued from a recurrent state up to k tokens old (silent wrong output).
- First greedy request after start no longer compiles the rejection kernels (`patch_spec_greedy_warmup.py`, vLLM #59260).
- Chat template (`runtime/inline_system_template.py`, applied at start): accepts `reasoning_effort` "high" (-> xhigh)
  and "minimal" (-> low) instead of HTTP 400 (Hermes and other OpenAI clients send "high"), and renders non-leading
  system messages in place so Claude Code's per-turn reminders on /v1/messages no longer break the prefix cache.
- `--enable-prompt-tokens-details`: responses report cached prompt tokens.
- Opt-in: `B70_MTP_K` (k=5 is +7-8% on short code/JSON but -4% at 64K), `B70_PREFIX_UNIT=128` (finer prefix-cache hits
  for agent turns, not yet validated), `B70_EAGER_PAD_TOKENS` (no measured gain).

## image b70.5 (2026-10-16): MTP decode ~10% faster on the dense 27B (no GPU sync in the GDN metadata builder)

Image `ghcr.io/riceharvest/b70-vllm:v0.31.0-b70.5` (= `latest`), same kernels as b70.4. New
`patches/b70/patch_gdn_spec_sync.py`, backported from vLLM main: on every MTP decode step the GDN metadata builder
gathered two GPU tensors with a host bool mask, and each of those gathers is a blocking index copy that waits for all
queued GPU work, so async scheduling lost its CPU/GPU overlap twice per step. The patch slices instead (same values).
Qwen3.8-27B GPTQ-Int4-asym, MTP k=4, greedy, 512-token answers, same session, b70.4 -> b70.5: code 101.5 -> 111.9,
json 110.9 -> 121.7, prose 60.0 -> 66.2, thinking 95.7 -> 107.0 tok/s (+10-12%), tokens per step unchanged
(4.13 / 2.25 / 3.6). The 30-vs-14 canary scores 5/6 on both. A second A/B on the v3 launcher stack: single-stream
93.3 -> 104.3 tok/s, ms per step 37.7 -> 33.8; 4 concurrent requests unchanged (230 tok/s).

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
