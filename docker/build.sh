#!/usr/bin/env bash
# Build (and optionally push) the prebuilt image. usage: docker/build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:-v0.31.0-b70.2}; IMG=ghcr.io/riceharvest/b70-vllm
[ -f kernels/libattn_kernels_xe_2.b70.so ] || gh release download v2 -p 'libattn_kernels_xe_2.b70.so' -D kernels
docker build -f docker/Dockerfile -t $IMG:$TAG -t $IMG:latest .
[ "${1:-}" = --push ] && docker push $IMG:$TAG && docker push $IMG:latest
