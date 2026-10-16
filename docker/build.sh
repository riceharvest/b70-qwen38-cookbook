#!/usr/bin/env bash
# Build (and optionally push) the prebuilt image. usage: docker/build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:-v0.31.0-b70.5}; IMG=ghcr.io/riceharvest/b70-vllm
[ -f kernels/libattn_kernels_xe_2.b70.so ] || gh release download v3 -p 'libattn_kernels_xe_2.b70.so' -D kernels
[ -f kernels/b70_moe_gemv.so ] || gh release download v3 -p 'b70_moe_gemv.so' -D kernels
# the patches must apply to the base image's vLLM before we build (CPU only, ~10 s; tests/README.md). SKIP_PATCH_TESTS=1 skips.
if [ "${SKIP_PATCH_TESTS:-0}" != 1 ]; then
  tests/fetch_vllm_src.sh --if-missing
  python3 -m pytest tests -q
fi
docker build -f docker/Dockerfile -t $IMG:$TAG -t $IMG:latest .
[ "${1:-}" = --push ] && docker push $IMG:$TAG && docker push $IMG:latest
