# Qwen3.8-27B on one Intel Arc Pro B70

I spent a couple of days squeezing Qwen3.8-27B onto a single Arc Pro B70 (32 GB), mostly with kernel work. This repo
has everything needed to reproduce it: launcher, vLLM changes (on fork branches), attention kernel patches (+ a prebuilt `.so`), the
quantizer, and the benchmark and quality scripts.

Made by [@imdariotoo](https://x.com/imdariotoo). I post my B70 and local-model releases on X, so follow there for the next ones.

What you get on one card, on the pinned `vllm/vllm-openai-xpu:v0.32.0` image:

- **262K context with vision**, a KV pool of ~320K tokens (fp8 KV), and MTP speculative decoding (k=4)
- **an asymmetric GPTQ int4 quant** at half the KL of a regular symmetric GPTQ at the same size
  ([stock](https://huggingface.co/darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym),
  [uncensored finetune](https://huggingface.co/darioooooo0o/Qwen3.8-27B-Uncensored-GPTQ-Int4-asym))
- fp8-KV prefill attention at fp16-KV speed (41 -> 72 TFLOPS), MTP verify that reads the KV cache once instead of
  5x, and an fp8 lm_head

Built on top of [SergiioB's B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook). Its MTP and
prefix-cache patches are carried on our vLLM fork branch (MIT). For a different engine, his EXL3 route is the other good option on this card.

New to the B70? [Awesome Arc Pro B70](https://github.com/riceharvest/awesome-b70) collects setup, engines and known driver bugs with workarounds. Setup problems (driver, firmware, Resizable BAR, permissions, GPU hangs)? Run [b70-doctor](https://github.com/riceharvest/b70-doctor) first: `curl -fsSL https://raw.githubusercontent.com/riceharvest/b70-doctor/main/b70-doctor.sh | bash` (read-only).

Want llama.cpp instead of vLLM? [b70-llama-builds](https://github.com/riceharvest/b70-llama-builds) has prebuilt llama.cpp for the B70: SYCL compiled ahead of time for the card with the oneAPI runtime bundled (no oneAPI install, no hour-long build), and Vulkan with the B70 fixes.

Progress per release (v1 to v3, before/after tables): [RESULTS.md](RESULTS.md).

**Where to look**

- Run it: [prebuilt image](#quick-start-prebuilt-image) or [from source](#quick-start-from-source)
- Single-user speed, long context, prefill and quality: [Numbers](#numbers)
- The 35B MoE: [Qwen3.6-35B-A3B](#qwen36-35b-a3b-moe-on-the-same-stack), its [byte budget](#how-fast-could-it-be-per-token-bytes-vs-608-gbs) and [64 concurrent users](#many-users-at-once-c64)
- What changed in the vLLM stack: [docs/patches.md](docs/patches.md); what each release gained: [RESULTS.md](RESULTS.md)
- Layout of this repo: [What's in here](#whats-in-here); dead ends: [Things I tried that didn't make it](#things-i-tried-that-didnt-make-it)

## Quick start (prebuilt image)

One command, no build. The image is the pinned vLLM XPU v0.32.0 with our B70 vLLM changes and the B70 attention
kernels already in it:

```bash
docker run -d --name b70 --device /dev/dri -v /dev/dri:/dev/dri:ro --shm-size 8g -p 8000:8000 \
  -v ~/models:/models -e MODEL=darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym ghcr.io/riceharvest/b70-vllm:latest
docker logs -f b70     # first start: model download, then ~5 min to load and capture graphs
```

OpenAI API on :8000, model name `qwen38`. Pick the model with `MODEL` (an HF repo id or a path inside the container):

| `MODEL` | what you get |
|---|---|
| `darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym` (default) | Qwen3.8-27B, 262K context + vision, MTP, tool calls |
| `darioooooo0o/Qwen3.8-27B-Uncensored-GPTQ-Int4-asym` | the uncensored finetune, same setup |
| `darioooooo0o/JEV-27B-VL-GPTQ-Int4-asym` | JEV decision model: `/v1/decide` (System 1) + chat (System 2), 131K context |
| `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` | Qwen3.6-35B-A3B MoE: ~120 tok/s without MTP, ~165-245 with it (see the MoE section) |

JEV repos are detected automatically (LoRA on, fp8 lm_head off). Env overrides: `PORT`, `SERVED_NAME`, `GPU_UTIL`,
`MAX_MODEL_LEN`, `FAST=0` (eager, no MTP). Anything after the image name is passed to vLLM. Packed MTP verify is on
(fixed v2 kernel, see below). Needs ~12 GB of free host RAM: embeddings and the vision tower live in host RAM via UVA.

Rebuild it yourself with `docker/build.sh` (Dockerfile in `docker/`). The vLLM changes are commits on fork branches
[riceharvest/vllm `b70/v0.32`](https://github.com/riceharvest/vllm/commits/b70/v0.32) and
[riceharvest/vllm-xpu-kernels `b70/v0.1.15.4`](https://github.com/riceharvest/vllm-xpu-kernels/commits/b70/v0.1.15.4),
pinned in `docker/fork.env`; the build copies the files they change over the stock image ([details](docs/patches.md)).

## Quick start (from source)

```bash
git clone https://github.com/riceharvest/b70-qwen38-cookbook && cd b70-qwen38-cookbook
hf download darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym --local-dir ~/models/qwen38-b70
# attention kernels for the pinned image (optional, but it's the fast path)
gh release download v3 -p '*.so' -D kernels   # attention kernels + MoE decode GEMV (or grab them from Releases)
MODEL_DIR=~/models/qwen38-b70 ./serve.sh                               # OpenAI API on :8000, model name "qwen38"
```

First start takes 2-3 min. `PORT`, `GPU_UTIL` (default 0.93) and `ZE_AFFINITY_MASK` are env overrides.

## Numbers

Stock Qwen3.8-27B quant, one B70 at the stock 275 W cap, one request at a time, 512-token answers. Spec decode speed
depends on the text, so the ms/step row (time per verify step) is the stable number.

| | short prompt | 32K | 65K | 100K |
|---|---|---|---|---|
| ms per verify step | 35 | 40 | 43 | 46 |
| decode tok/s, greedy | code 113 / json 119 / prose 66 / reasoning 96 | 72 | 72 | 65-67 |
| decode tok/s, sampled (T=1, top_p .95, top_k 20) | code 107-111 / json 115-122 / prose 60-66 / reasoning 92-97 | - | 73-77 | 66-69 |

Image b70.5 (2026-10-16) decodes another ~10% faster with MTP: same session, b70.4 -> b70.5, code 101.5 -> 111.9,
json 110.9 -> 121.7, prose 60.0 -> 66.2, reasoning 95.7 -> 107.0 tok/s, same tokens per step ([RESULTS](RESULTS.md)).
Image b70.6 (2026-10-16) is 3.5-6% faster again, most at long context (64K: 42.8 -> 40.3 ms per step), fixes a
stale-state bug on MTP steps without drafts, and accepts `reasoning_effort: high` ([RESULTS](RESULTS.md)).

| Prefill (cold) | 5.7K | 24K | 48K | 74K | 237K |
|---|---|---|---|---|---|
| tok/s | 2042 | 1822 | 1609 | 1422 | 824 (needle found) |

- Prefix cache: a 16K-token prompt takes 8.5 s cold and 1.5 s warm.
- **Packed MTP verify: fixed in the v2 kernel (2026-10-15).** With the v1 `.so`, a split whose keys were all masked
  for draft row 0 gave a softmax sum of 0, the epilogue computed 0 x (1/0) = NaN, and the output turned into `!!!!`.
  That happened at specific KV lengths (e.g. 2049-2052) with 1-4 requests in flight. The v2 kernel adds a zero-sum guard
  (`kernels/zero-sum-guard.patch`). Kernel A/B: NaN in exactly the 28 predicted cases with v1, 0 with v2. Serving,
  8 concurrent thinking-length answers: v1 2-3/16 corrupted per run, v2 0/48. Packed is on by default again;
  `serve.sh` keeps it off if it finds the v1 `.so`. Single stream at 131K ctx: packed 82 / 77 tok/s (code / prose)
  vs 80 / 77 off.
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

## Qwen3.6-35B-A3B (MoE) on the same stack

Same image, different `MODEL`. Text-only use: add `--language-model-only` after the image name (it frees the vision
profiling memory):

```bash
docker run -d --name b70 --device /dev/dri -v /dev/dri:/dev/dri:ro --shm-size 8g -p 8000:8000 -v ~/models:/models \
  -e MODEL=palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 ghcr.io/riceharvest/b70-vllm:latest --language-model-only
```

That checkpoint ships the routed experts in int4 but attention, linear attention (GDN), the shared expert and the
lm_head in bf16. Two MoE-specific additions make the difference on a B70:

- **int4 MoE decode GEMV** (`kernels/moe-gemv/`, fork commit `patch_moe_gemv`): at decode each expert sees one row,
  and the stock CUTLASS grouped GEMM spends ~40 us per call no matter how little work there is (8 experts = 13 MB, ~21 us
  at full bandwidth). The GEMV takes over when a call has <= 8 rows: gate_up 56 -> 18 us, down 34 -> 14 us per layer.
  Bigger calls (MTP verify, prefill) stay on CUTLASS.
- **fp8 weight-only for the linears a checkpoint ships in bf16** (fork commit `patch_linear_fp8`): attention, GDN
  projections and the shared expert are read as fp8 e4m3 with per-row scales (same kernel as the fp8 lm_head). KL vs the
  bf16 weights 0.019, top-1 agreement 95.2%. `B70_LINEAR_FP8=0` turns it off. Since image b70.4 the bf16 copy is freed
  (weights 19.13 instead of 21.75 GiB, same speed); on b70.2 freeing it faulted the GPU under MTP, on b70.3 that no longer
  reproduces (probes, MTP4 under load with and without prefix caching, 10 min mixed-load stress). `B70_LINEAR_FP8_FREE=0`
  keeps the copy if you ever see a fault.

Single request, 16K context, `GPU_UTIL=0.82`, code / prose answers of 512 tokens, decode tok/s:

| setup | no MTP | MTP k=4 |
|---|---|---|
| stock vLLM XPU v0.31 image | 75.6 / 76.3 | 123.4 / 113.2 (fits <=25 GB only at 2K context) |
| + cookbook patches (int4 draft head, int4 MTP layer, packed verify) | 73.4 / 73.7 | 175.9 / 126.6 |
| + fp8 lm_head | 78.7 / 79.1 | 183.9 / 134.3 |
| + int4 MoE decode GEMV | 93.8 / 94.3 | 200.8 / 145.4 |
| **+ fp8 attention/GDN/shared expert (the image default)** | **118.9 / 119.6** | **243.7 / 164.4** |

Prefill ~9.3-10.3K tok/s. MTP acceptance 3.3-3.6 on code, ~2.6 on prose. 0/32 corrupted outputs with 8 concurrent
thinking-length requests. For comparison on localmaxxing (single B70, one request): 248 tok/s (vLLM 0.26 + MTP4, same
checkpoint), 205 (vLLM 0.21 + MTP4), and the best llama.cpp runs I found are ~132 (SYCL, fused kernels, no MTP).

**Many users at once.** For batch serving add `--mamba-ssm-cache-dtype float16`: Qwen3.5/3.6/3.8 store their linear-attention
state in fp32 by default (60 MiB per request on this model), which at 64 requests is about a third of what the GPU reads
per step. With it, 64 concurrent users get 42-45 ms per token (generation 1420-1510 tok/s over two runs, 2581-2769 tok/s total
with prompts) instead of 49.3 ms with only 61 of 64 running; decode quality stays at the fp32 noise floor and gsm8k is 98/100
(fp32: 94/100)
([numbers, quality check, exact command](https://riceharvest.github.io/awesome-b70/results/vllm-64-users.html)). With
`serve.sh`: `B70_GDN_STATE=fp16 ./serve.sh`; with the image: `-e B70_GDN_STATE=fp16` (or `--mamba-ssm-cache-dtype float16`).
Keep the default `fp32` state for long-context (32K-262K) single-stream runs: the Xe2 chunk prefill kernel writes the state
back between 64-token chunks, so an `fp16` state cache can overflow 65504 on long prompts ([vllm#62781](https://github.com/vllm-project/vllm/issues/62781)).
Quality was checked without MTP; with MTP the per-draft state slots are fp16 too (not measured yet).

**Decode GEMV check.** It matches CUTLASS to 3e-4 relative error per call (stress test: 900 routings incl. outlier
activations; 500 replays inside an XPU graph), and in serving an eager cross-check of every decode call found zero
mismatches over 2,048 steps. Greedy decode GEMV-vs-CUTLASS: per-step KL p50 0.0001, p99 0.043, 14/1555 steps > 0.05
vs 12/1193 for CUTLASS graph-vs-eager (the numeric noise floor of the same code); divergences sit at near-ties.

### How fast could it be? (per-token bytes vs 608 GB/s)

| GB per generated token | as shipped (bf16 attn/GDN/head) | all int4 | int4 + fp8 head | int4 + int4 head |
|---|---|---|---|---|
| routed experts (8 of 256, int4) | 0.53 | 0.53 | 0.53 | 0.53 |
| shared expert | 0.25 | 0.065 | 0.065 | 0.065 |
| router | 0.04 | 0.04 | 0.04 | 0.04 |
| attention (10 layers) | 0.55 | 0.14 | 0.14 | 0.14 |
| linear attention / GDN (30 layers) | 2.02 | 0.52 | 0.52 | 0.52 |
| lm_head (248K x 2048) | 1.02 | 1.02 | 0.51 | 0.26 |
| GDN state + KV at 2K + embedding row | ~0.12 | ~0.12 | ~0.12 | ~0.12 |
| **total** | **4.53** | **2.44** | **1.93** | **1.68** |
| no MTP, tok/s at 100% / 88% of 608 GB/s | 134 / 118 | 249 / 219 | 315 / 277 | 362 / 318 |
| MTP k=4 step (experts x4.7 + rest + 4 drafts) | 7.17 GB | 5.08 GB | 4.57 GB | 4.32 GB |
| MTP k=4 at acceptance 3.66 (code) / 2.66 (prose), 100% | 310 / 226 | 438 / 318 | 487 / 354 | 515 / 374 |

The image default reads fp8 attention/GDN/shared + fp8 head, about 2.6 GB per token: a ~233 tok/s ceiling without MTP,
of which 119 is 51%. The rest is the routing and GDN kernels and the GEMV's own overhead. The x4.7 for experts under MTP
assumes independent routing per draft token (5 tokens touch ~38 experts); real routing overlaps, so it's pessimistic.

**What 800 tok/s single-stream would take:** at most 0.76 GB read per generated token. Without MTP that is impossible:
the leanest layout above reads 1.68 GB. With MTP k=4 at 3.66 accepted tokens a step may read ~2.8 GB, but the experts
touched by 5 tokens alone are ~2.5 GB. It would need ~2-bit weights everywhere plus k>=8 drafts with near-perfect
acceptance. 800+ tok/s is real only as aggregate throughput over many concurrent requests.

### Many users at once (c=64)

64 concurrent users, random 512-token prompts, 512 output tokens each (`ignore_eos`), 16K max context, no MTP, one B70,
VRAM capped at <=25 GB (`--gpu-memory-utilization 0.82`). Compared with the best public B70 entry on
[localmaxxing](https://localmaxxing.com) for this model at batch 64 (SergiioB, 2026-08-07, same workload shape):

| | SergiioB (localmaxxing) | this stack | |
|---|---|---|---|
| generation tok/s (= 64 / median TPOT) | 1139.8 (TPOT 56 ms) | **1511.5** (TPOT 42.3 ms, p99 45.3) | +33% |
| total tok/s (prompt + output, wall clock) | 1967.7 | **2768.9** | +41% |
| requests actually running | 64 | 64 (engine log: Running 64, Waiting 0, KV cache 54%) | |
| rerun: the command below, as written | | 1420 generation (TPOT 45.1 ms) / 2581 total, Running 64 | +25% / +31% |

The rerun came ~1 h later on a warmer card (the whole box measured ~4% slower that afternoon in other tests too), so
expect roughly 1420-1510 generation and 2580-2770 total.

32 users: 1081.8 generation / 1986 total. What made the difference at c=64:

- **GDN (linear-attention) recurrent state in fp16** instead of the checkpoint's fp32 (`--mamba-ssm-cache-dtype float16`).
  At 64 sequences the state is ~1/3 of the bytes moved per decode step; fp16 halves it and halves its memory, so all
  64 requests fit. Quality check on the decode path (8 x 1024-token greedy generations, each scored against fp32-state
  prefill): KL 0.0027 / 0.0034 / 0.0035 at positions 0-255 / 256-511 / 512+, versus 0.0049 / 0.0023 / 0.0041 for
  the fp32 state itself (the numerical noise floor); top-1 agreement 98.6-99.5%, same as fp32. gsm8k (first 100):
  98/100 with the fp16 state, 94/100 with fp32 (same server flags; the gap is within run noise for 100 questions).
- **Prefix caching off.** With it on, vLLM keeps extra GDN state per block ("align" mode) and only 29 of 64 requests fit.
- fp8 weight-only attention/GDN/shared expert + fp8 lm_head (the image default; KL 0.019 vs the checkpoint), with the
  bf16 copies freed (`B70_LINEAR_FP8_FREE=1`, the default since b70.4).

Caveats: SergiioB's checkpoint (llmfan46's MTP-preserved GPTQ-Int4) also stores the linear-attention projections and the
shared expert in int4, so it reads fewer bytes per step than palmfuture's (bf16 there, fp8 here); that favours his run.
His was measured at a 165 W power cap, ours at the card's stock power limit (draw not logged). Different vLLM (his
0.21 build, ours v0.31: the numbers above are image b70.3; the command now uses b70.7, which I haven't re-measured at
c=64 yet). MTP doesn't help here yet: each speculative token keeps its own fp32 GDN state slot, so MTP
at c=64 doesn't fit in 25 GB.

```bash
docker run -d --name b70-many --device /dev/dri -v /dev/dri:/dev/dri:ro --shm-size 8g -p 8000:8000 -v ~/models:/models \
  -e B70_TARGET_LMHEAD_FP8=1 -e B70_LINEAR_FP8_FREE=1 \
  -e 'B70_LINEAR_FP8_RE=(self_attn\.(qkv_proj|o_proj)|linear_attn\.(in_proj_qkvz|in_proj_qkv|in_proj_z|out_proj)|shared_expert\.(gate_up_proj|down_proj))$' \
  --entrypoint vllm ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7 serve palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 \
  --served-model-name moe --port 8000 --distributed-executor-backend mp --quantization gptq --dtype float16 \
  --kv-cache-dtype fp8 --max-model-len 16384 --gpu-memory-utilization 0.82 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --language-model-only --cpu-offload-gb 4 --cpu-offload-params embed_tokens --no-enable-prefix-caching \
  --mamba-ssm-cache-dtype float16 --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64]}'

# the benchmark (vLLM's own client, inside the container)
docker exec b70-many vllm bench serve --backend vllm --base-url http://localhost:8000 --model moe \
  --tokenizer palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 --dataset-name random --random-input-len 512 --random-output-len 512 \
  --ignore-eos --num-prompts 128 --max-concurrency 64 --seed 0 --temperature 0 --percentile-metrics ttft,tpot,itl
```

## What's in here

| Path | |
|---|---|
| `serve.sh` | the launcher (pinned image digest, the fork overlay, flags) |
| `kernels/moe-gemv/` | int4 MoE decode GEMV (SYCL source + build.sh); the built `b70_moe_gemv.so` is in release v3 |
| `docker/` | the prebuilt image: `Dockerfile`, entrypoint `b70-serve.sh` (picks chat or JEV setup from the model), `build.sh`, `fork.env` + `overlay.sh` (the vLLM fork pins and their export) |
| `docs/patches.md` | the vLLM changes on the fork branches: asym GPTQ on XPU, fp8 lm_head, packed MTP verify hook, embed UVA, pruned draft vocab, SergiioB's MTP / GDN / prefix-cache fixes |
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

My code: MIT. SergiioB's ported patches: MIT (`licenses/SergiioB.LICENSE`). Kernel patches and `.so`: under their upstreams' licenses
(Apache-2.0 / BSD-3-Clause). Model weights: Apache-2.0, inherited from Qwen.
