# Prebuilt B70 image: pinned vLLM XPU v0.31.0 + the cookbook's patches applied + the B70 attention kernels baked in.
# Build from the repo root (needs kernels/libattn_kernels_xe_2.b70.so; docker/build.sh fetches it from the v2 release):
#   docker build -f docker/Dockerfile -t ghcr.io/riceharvest/b70-vllm:v0.31.0-b70.5 .
FROM vllm/vllm-openai-xpu:v0.31.0@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7
LABEL org.opencontainers.image.source="https://github.com/riceharvest/b70-qwen38-cookbook" \
      org.opencontainers.image.description="vLLM XPU v0.31.0 for Intel Arc Pro B70: cookbook patches + B70 attention kernels baked in" \
      org.opencontainers.image.licenses="MIT"
COPY patches /b70/patches
COPY runtime /opt/b70
COPY kernels/b70_moe_gemv.so /opt/b70moe/b70_moe_gemv.so
COPY kernels/libattn_kernels_xe_2.b70.so /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels_xe_2.so
# All patches are switched by env vars at runtime, so applying them here changes nothing until a B70_* var turns them on.
RUN set -e; \
    for p in patch_mtp_nightly patch_mtp_boundary patch_draft_lmhead_int4 patch_draft_mtp_int4 patch_gdn_mixed_split_v5 \
             patch_fix_accepted_sync patch_fix_backward_copy patch_fix_eagle_drop; do python /b70/patches/sergiiob/$p.py; done; \
    for p in patch_embed_uva patch_draft_vocab patch_asym_gptq patch_target_lmhead_fp8 patch_spec_packed patch_moe_gemv patch_linear_fp8 patch_gdn_spec_sync patch_gdn_zero_draft patch_gdn_mixed_no_sync patch_xpu_graph_kv_splits; do python /b70/patches/b70/$p.py; done
COPY docker/b70-serve.sh /usr/local/bin/b70-serve
ENV PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_TARGET_DEVICE=xpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0 \
    CCL_ATL_TRANSPORT=mpi CCL_PROCESS_LAUNCHER=none B70_MTP_BF16_DRAFT=1 B70_DRAFT_LMHEAD_INT4=1 B70_DRAFT_MTP_INT4=1 \
    VLLM_XPU_ENABLE_XPU_GRAPH=1 B70_DRAFT_VOCAB=/opt/b70/draft_vocab_64k.json B70_SPEC_PACKED=1 B70_MOE_GEMV=1 HF_HOME=/models
ENTRYPOINT ["b70-serve"]
