# Notes: what worked, what didn't

All numbers are from one Arc Pro B70 at 275 W, the `vllm/vllm-openai-xpu:v0.31.0` image, Qwen3.8-27B (hybrid: 48 GDN +
16 full-attention layers, head_dim 256, GQA 24:4), fp8 KV, page 64, MTP k=4, single request.

## Quantization
- On this model, symmetric GPTQ int4 g128 is the problem, not the runtime. The int4 GEMM matches an fp32 reference
  dequant (rel err 2.7e-4), and the canary fails the same way with bf16 activations, fp16/fp8 KV, or MTP off.
- **Asymmetric int4 + MSE clip search** (11 clip ratios per group, error^2.4) inside GPTQ roughly halves KL vs BF16 at
  the same size and speed. Stock XPU loading of asym GPTQ is broken: it transposes qzeros. `patch_asym_gptq.py` fixes that.
- GPTQ v1 can't store zero point 0 (it stores z-1). With asym + MSE that happened once in 190M groups. The export
  bumps that group to z=1, which changes exactly one weight.
- `b70quant.py` streams one layer at a time on the B70 (~1 h for 64 layers, 128 x 2048 calibration tokens).
- Measure quality as KL vs BF16 on held-out windows (top-20 logits), not perplexity. Canaries catch the worst
  defects but are noisy.

## Kernels
- fp8 KV prefill attention ran at 41 TFLOPS vs 72 with fp16 KV. The cause was the e4m3 -> half conversion (11 instructions
  per 64 values), re-done for every subgroup along Q with head 256. Three instructions plus folding x256 into the
  softmax/V scales gets it to parity, bit-identical on all 254 non-NaN codes.
- Stock MTP verify (q_len 5) runs the decode kernel once per position, so it reads the KV cache 5x.
  Packing positions x GQA heads into rows with a per-row causal mask reads it once: 65K verify 671 -> 565 us/layer.
- Routing verify through chunk-prefill instead is 3-9x worse.
- A 32-row single-workgroup verify tile was correct and 12-14% faster in a microbenchmark but slower in serving
  (+1-2 ms/step). Probably the 4 subgroups each reload K/V. The next step would be a K/V tile shared in SLM.
- The fp16 lm_head (248K x 5120) costs 4.3 ms per verify step. As per-row fp8 it costs 2.2 ms, KL +0.0006.
- XPU graphs with a small capture set ([1..40]) cost 0.3 GiB instead of 3.7 GiB, and decode gets 5-19% faster.

## Spec decode
- k=3/4/5 are within noise; k=4 is the default. Keep the draft MTP layer and draft head in int4 (a BF16 draft is 25-30% slower).
- Probabilistic drafting (`draft_sample_method: probabilistic`) gives +3% tokens/step at T=1, and it's lossless.
  Greedy outputs can differ at numeric near-ties (I saw a 0.03-nat one), but no quality loss.
- The pruned 64K draft vocab is lossless (the target verifies against the full vocab) and makes the draft head cheaper.
- Compare decode speed in ms per verify step. Sampled tokens/step moves ~10% between restarts.

## Memory / context
- Keeping embed_tokens and the vision tower in host RAM via UVA frees ~3.2 GiB of VRAM at no measurable speed cost.
- At `--gpu-memory-utilization 0.93` that leaves a ~320K-token fp8 KV pool. A 241K-token needle was retrieved correctly.
- Running prefix caching + MTP without the accepted_sync / backward_copy / eagle_drop fixes risks silent corruption.
  They're included.

## Building
- Restrict AOT to the card: `VLLM_XPU_XE2_AOT_DEVICES=bmg-g31-a0`. That takes the build from ~1 h to ~8.5 min. Device
  codegen at link time is serial per target, so more `-j` doesn't help.
- icpx + CUTLASS templates: ~8 GB per job. `-j16` OOM-killed a 64 GB desktop. Use `-j2`, a `--memory` cap, and a watchdog.
- Benchmark with the minimum of several rounds. The card often also drives the display, and 20-iteration means were
  +-2x noisy.
- Never `pkill -f` with a pattern that appears in your own command line.
