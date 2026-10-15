| Patch | What it does |
|---|---|
| `patch_embed_uva.py` | keeps `embed_tokens` in host RAM via UVA. Together with `--cpu-offload-params embed_tokens visual` (vision tower in host RAM too), that frees ~3.2 GiB of VRAM for KV cache. No measurable speed loss on a B70; vision still works. |
| `patch_draft_vocab.py` | the MTP draft head only scores a 64K-token subset (`runtime/draft_vocab_64k.json`). Lossless: the target model verifies with the full vocab. |
| `patch_asym_gptq.py` | loads asymmetric GPTQ (`sym: false`, uint4 + zero points) on XPU: adds the GPTQ v1 +1 back at load and stops the stock XPU path from transposing qzeros (a bug in stock). |
| `patch_target_lmhead_fp8.py` | converts the 2.5 GB fp16 lm_head to per-row fp8 e4m3 at load (`fp8_gemm_w8a16`). Head read per step 4.3 -> 2.2 ms, +28K KV tokens, KL cost +0.0006. The int4 draft head is built from the fp16 weights first. |
| `patch_spec_packed.py` | imports `runtime/spec_packed.py`, which routes MTP verify (q_len 5) to the packed per-row-mask decode kernel. Needs the B70 `.so`. |
| `patch_gdn_spec_sync.py` | backport from vLLM main: the GDN metadata builder's pure MTP-decode branch slices the state indices and `num_accepted_tokens` instead of gathering them with a host bool mask. Each mask gather was a blocking H2D index copy, i.e. a full GPU queue drain, twice per MTP decode step. Same values. Runs after `sergiiob/patch_mtp_boundary.py`. |
