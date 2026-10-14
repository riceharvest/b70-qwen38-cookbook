# Qwen3.8-27B on one Intel Arc Pro B70

I spent a couple of days squeezing Qwen3.8-27B onto a single Arc Pro B70 (32 GB), mostly with kernel work. This repo
has everything needed to reproduce it: launcher, vLLM patches, attention kernel patches (+ a prebuilt `.so`), the
quantizer, and the benchmark and quality scripts.

What you get on one card, on the pinned `vllm/vllm-openai-xpu:v0.31.0` image:

- **262K context with vision**, a KV pool of ~320K tokens (fp8 KV), and MTP speculative decoding (k=4)
- **an asymmetric GPTQ int4 quant** at half the KL of a regular symmetric GPTQ at the same size
  ([stock](https://huggingface.co/darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym),
  [uncensored finetune](https://huggingface.co/darioooooo0o/Qwen3.8-27B-Uncensored-GPTQ-Int4-asym))
- fp8-KV prefill attention at fp16-KV speed (41 -> 72 TFLOPS), MTP verify that reads the KV cache once instead of
  5x, and an fp8 lm_head

Built on top of [SergiioB's B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook). Its MTP and
prefix-cache patches are included here (MIT). For a different engine, his EXL3 route is the other good option on this card.

## Quick start

```bash
git clone https://github.com/riceharvest/b70-qwen38-cookbook && cd b70-qwen38-cookbook
hf download darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym --local-dir ~/models/qwen38-b70
# attention kernels for the pinned image (optional, but it's the fast path)
gh release download v1 -p 'libattn_kernels_xe_2.b70.so' -D kernels   # or grab it from the Releases page
MODEL_DIR=~/models/qwen38-b70 ./serve.sh                               # OpenAI API on :8000, model name "qwen38"
```

Needs Docker with `/dev/dri`, one B70, and ~12 GB of free host RAM (embeddings and the vision tower live in host RAM
via UVA). First start takes 2-3 min. `PORT`, `GPU_UTIL` (default 0.93) and `ZE_AFFINITY_MASK` are env overrides.

## Numbers

Stock Qwen3.8-27B quant, one B70 at the stock 275 W cap, one request at a time, 512-token answers. Spec decode speed
depends on the text, so the ms/step row (time per verify step) is the stable number.

| | short prompt | 32K | 65K | 100K |
|---|---|---|---|---|
| ms per verify step | 35 | 40 | 43 | 46 |
| decode tok/s, greedy | code 113 / json 119 / prose 66 / reasoning 96 | 72 | 72 | 65-67 |
| decode tok/s, sampled (T=1, top_p .95, top_k 20) | code 107-111 / json 115-122 / prose 60-66 / reasoning 92-97 | - | 73-77 | 66-69 |

| Prefill (cold) | 5.7K | 24K | 48K | 74K | 237K |
|---|---|---|---|---|---|
| tok/s | 2042 | 1822 | 1609 | 1422 | 824 (needle found) |

- Prefix cache: a 16K-token prompt takes 8.5 s cold and 1.5 s warm.
- **Packed MTP verify is now off by default (fixed 2026-10-14).** With about 8 long generations in flight it turned
  some outputs into `!!!!` garbage (2 of 16 thinking-length answers, 0 of 32 with it off). It only saves about 3% for
  a single request (code 84 vs 81 tok/s, prose 78 vs 76 on my quick bench), so it isn't worth it. The decode numbers
  above were measured with it on; `B70_SPEC_PACKED=1` turns it back on for single-user setups.
- Short requests: with 12 concurrent, 10-11 of 12 outputs are byte-identical to running them alone. The rest diverge
  at near-ties (batched fp16 math isn't batch-invariant).

**Quality**: KL divergence vs BF16 over the top-20 logits on held-out 2048-token windows. Lower is better.

| stock Qwen3.8-27B | code | technical prose | chat (private samples) | all | top-1 agree |
|---|---|---|---|---|---|
| [SergiioB's GPTQ int4](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) (sym g128, GPTQModel) | 0.087 | 0.103 | 0.050 | 0.078 | 91.3% |
| **this quant** (asym g128 + MSE clip) | **0.037** | **0.057** | **0.025** | **0.039** | **93.8%** |
| this quant + fp8 lm_head (as served) | 0.038 | 0.058 | 0.025 | 0.039 | 93.5% |

The cookbook's 6-question arithmetic canary: mine scores 4/6, SergiioB's 6/6. With 6 greedy near-tie questions that's
noise. On 100 seeded questions of the same kind (`bench/arith_battery.py`), mine scores 48/100 and his 49/100.

## What's in here

| Path | |
|---|---|
| `serve.sh` | the launcher (pinned image digest, all patches, flags) |
| `patches/b70/` | my vLLM patches: asym GPTQ on XPU, fp8 lm_head, packed MTP verify hook, embed UVA, pruned draft vocab ([details](patches/b70/README.md)) |
| `patches/sergiiob/` | SergiioB's MTP / GDN / prefix-cache patches, unchanged ([source](patches/sergiiob/SOURCE.md)) |
| `runtime/` | packed-verify dispatcher and the 64K draft vocab, mounted at `/opt/b70` |
| `kernels/` | attention kernel patches, kernel configs, `build.sh`, and how the `.so` was built ([details](kernels/README.md)) |
| `quant/` | `b70quant.py` (GPTQ asym int4 g128 + MSE clip search, streams one layer at a time on the B70), GPTQ v1 export + independent check, calib/vocab builders, BF16 reference + KL scorer |
| `bench/` | serving harness, canary, needle, concurrency, vision, and attention microbenchmarks |
| `docs/NOTES.md` | what worked, what didn't, and gotchas |

## Things I tried that didn't make it

- **32-row verify tile:** 12-14% faster in isolation, but slower inside the real server. Opt-in via `B70_SPEC_Q32=1`.
- **W4A8 for prefill:** ~5-10% faster cold prefill for a real KL cost.
- **Block-wise spec verification:** within noise.
- **Routing MTP verify through chunk-prefill:** 3-9x worse.

## Upstream

The fp8 KV conversion is opt-in upstream as [intel/sycl-tla#859](https://github.com/intel/sycl-tla/pull/859) +
[vllm-project/vllm-xpu-kernels#638](https://github.com/vllm-project/vllm-xpu-kernels/pull/638).

## License

My code: MIT. `patches/sergiiob/`: MIT (SergiioB). Kernel patches and `.so`: under their upstreams' licenses
(Apache-2.0 / BSD-3-Clause). Model weights: Apache-2.0, inherited from Qwen.
