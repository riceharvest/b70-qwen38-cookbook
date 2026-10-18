# Prebuilt B70 image: pinned vLLM XPU v0.32.0 + the B70 fork changes (docker/fork.env) + the B70 attention kernels baked in.
# Build from the repo root with docker/build.sh (it exports build/overlay from the forks and fetches the kernels), or:
#   docker/overlay.sh &&
#   docker build -f docker/Dockerfile -t ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7 .
FROM vllm/vllm-openai-xpu:v0.32.0@sha256:20509fabe50dda62c09c0786d1a6effa299d3dbdddcc49331b93e65144652a8c
LABEL org.opencontainers.image.source="https://github.com/riceharvest/b70-qwen38-cookbook" \
      org.opencontainers.image.description="vLLM XPU v0.32.0 for Intel Arc Pro B70: B70 fork changes + B70 attention kernels baked in" \
      org.opencontainers.image.licenses="MIT"
COPY runtime /opt/b70
COPY kernels/b70_moe_gemv.so /opt/b70moe/b70_moe_gemv.so
COPY kernels/libattn_kernels_xe_2.b70.so /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels_xe_2.so
# The fork files replace whole files (tests/test_overlay.py checks the image's originals equal the fork base). Every change
# is switched by an env var at runtime, so it changes nothing until a B70_* var turns it on.
COPY build/overlay/site-packages/ /opt/venv/lib/python3.12/site-packages/
COPY build/overlay/removed.txt build/overlay/REFS /b70/overlay/
RUN cd /opt/venv/lib/python3.12/site-packages && xargs -r rm -- < /b70/overlay/removed.txt
COPY docker/b70-serve.sh /usr/local/bin/b70-serve
ENV PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_TARGET_DEVICE=xpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0 \
    CCL_ATL_TRANSPORT=mpi CCL_PROCESS_LAUNCHER=none B70_MTP_BF16_DRAFT=1 B70_DRAFT_LMHEAD_INT4=1 B70_DRAFT_MTP_INT4=1 \
    VLLM_XPU_ENABLE_XPU_GRAPH=1 B70_DRAFT_VOCAB=/opt/b70/draft_vocab_64k.json B70_SPEC_PACKED=1 B70_MOE_GEMV=1 HF_HOME=/models
ENTRYPOINT ["b70-serve"]
