# B70 vLLM changes

The changes live as one commit each on two fork branches, pinned in `docker/fork.env`:

- [riceharvest/vllm `b70/v0.32`](https://github.com/riceharvest/vllm/commits/b70/v0.32): vLLM v0.32.0 + 18 commits
- [riceharvest/vllm-xpu-kernels `b70/v0.1.15.4`](https://github.com/riceharvest/vllm-xpu-kernels/commits/b70/v0.1.15.4):
  the Python side of `patch_spec_packed` and `patch_moe_gemv` on the source of the 0.1.15.4 wheel

`docker/overlay.sh` exports the files they change into `build/overlay`; the Dockerfile and `serve.sh` copy them over the
stock image. Each commit is named after the patch script it replaced (`[B70] patch_<name>`); every change is off until
its `B70_*` switch turns it on. To work on one: edit on the branch, push, bump the SHA in `docker/fork.env`.
New vLLM release: `git rebase --onto vX.Y.Z v0.32.0` into a new branch `b70/vX.Y`, bump the base and ref, and run the
tests (`tests/test_overlay.py` fails if the image's files differ from the fork base).

| Change | What it does |
|---|---|
| `patch_embed_uva` | keeps `embed_tokens` in host RAM via UVA. Together with `--cpu-offload-params embed_tokens visual` (vision tower in host RAM too), that frees ~3.2 GiB of VRAM for KV cache. No measurable speed loss on a B70; vision still works. |
| `patch_draft_vocab` | the MTP draft head only scores a 64K-token subset (`runtime/draft_vocab_64k.json`). Lossless: the target model verifies with the full vocab. |
| `patch_asym_gptq` | loads asymmetric GPTQ (`sym: false`, uint4 + zero points) on XPU: adds the GPTQ v1 +1 back at load and stops the stock XPU path from transposing qzeros (a bug in stock). |
| `patch_target_lmhead_fp8` | converts the 2.5 GB fp16 lm_head to per-row fp8 e4m3 at load (`fp8_gemm_w8a16`). Head read per step 4.3 -> 2.2 ms, +28K KV tokens, KL cost +0.0006. The int4 draft head is built from the fp16 weights first. |
| `patch_spec_packed` | imports `runtime/spec_packed.py`, which routes MTP verify (q_len 5) to the packed per-row-mask decode kernel. Needs the B70 `.so`. |
| `patch_gdn_mixed_no_sync` | vLLM v0.32 no longer syncs on pure MTP-decode steps; steps that mix MTP decodes with a prefill build the GDN row indices and cumsums on the host and send them in one async copy, instead of six host-mask gathers (each a full GPU queue drain). Same values. The row indices travel in the metadata, so the other GDN KV-cache groups (`update_block_table`, 3 groups on Qwen3.8-27B) gather with them too; on v0.31 those still synced 4 times per mixed step. `B70_GDN_MIXED_NO_SYNC=0` switches it off. Not yet GPU-tested. |
| `patch_xpu_graph_kv_splits` | under XPU-graph capture `max_seqlen_k` is `max_model_len`, so the decode kernel replays its long-context split count at every context length. q=1 decode (MTP draft steps) now uses one work-group per Xe core; `runtime/spec_packed.py` does the same for the packed verify rows (4 work-groups per KV head at head_dim 256, so 2 splits at batch 1, 1 at batch >= 2, vs 32/17/8/4 before). Eager calls unchanged. `B70_GRAPH_KV_SPLITS=0` switches both off. Not yet GPU-tested. |
| `patch_spec_greedy_warmup` | backport of vLLM #59260: a second startup warmup pass with greedy requests, so the rejection-sampling kernels' fp16 variants compile at startup instead of in the first greedy request (seen in our server log as "Triton kernel JIT compilation during inference"). Startup only. |
| `patch_eager_token_pad` | eager steps (prefill chunks, prefill + decodes) can round their token count up to a multiple of `B70_EAGER_PAD_TOKENS` (opt-in; default 0 = off since the serving A/B showed no TTFT change). The oneDNN int4 GEMM caches one primitive per exact token count in a 512-entry LRU shared by all int4 layers, and building one costs 20-50 ms per weight shape on the B70, so nearly every new prompt length stalled its step by ~0.1-0.3 s. Padded rows write no KV and are never sampled. `B70_EAGER_PAD_TOKENS=0` switches it off. Not yet GPU-tested. |

The MoE GEMV and fp8 weight-only linears (`patch_moe_gemv`, `patch_linear_fp8`) are described in the main README.

Dropped on the v0.32.0 base (image b70.7), because vLLM v0.32 does the same upstream: `patch_gdn_spec_sync` (pure-spec
slicing), `patch_gdn_zero_draft` (#56531 + #55504) and `sergiiob/patch_mtp_boundary` (v0.32 keeps a partial final
group on the speculative path with its real token count). The `B70_GDN_ZERO_DRAFT` and `B70_MTP_PARTIAL_FINAL_GROUP`
switches are gone with them. `tests/test_overlay.py::test_base_covers_dropped_gdn_patches` fails if a base image lacks
them.

## SergiioB's changes

The commits marked `Co-authored-by: SergiioB` (`patch_mtp_nightly`, `patch_draft_lmhead_int4`, `patch_draft_mtp_int4`,
`patch_gdn_mixed_split_v5`, `patch_fix_accepted_sync`, `patch_fix_backward_copy`, `patch_fix_eagle_drop`) port his
patches from [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
at commit `1a76136ea1b3f96bba3b20e42041a6fcc95f56fd`, MIT License (`licenses/SergiioB.LICENSE`, also on the fork
branch). They add MTP spec decode fixes for Qwen3.8 on vLLM XPU (nightly MTP), the int4 draft MTP layer and draft
lm_head, the GDN mixed-batch split, and three prefix-cache x MTP correctness fixes (accepted_sync, backward_copy,
eagle_drop). Thanks, SergiioB.
