#!/usr/bin/env bash
# Copy the .py files of vllm + vllm_xpu_kernels out of the pinned base image (the Dockerfile's FROM), for
# tests/test_patches.py. No GPU, no network inside the container. ~40 MB. usage: tests/fetch_vllm_src.sh [dest]
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=$(sed -n 's/^FROM \([^ ]*\).*/\1/p' docker/Dockerfile | head -1)
DEST=${1:-${B70_VLLM_SRC:-$HOME/.cache/b70-cookbook-tests/vllm-xpu-v0.31.0/site-packages}}
mkdir -p "$DEST"
docker run --rm --network none --entrypoint bash "$IMG" -c \
  "cd /opt/venv/lib/python3.12/site-packages && find vllm vllm_xpu_kernels -name '*.py' | tar -cf - -T -" | tar -xf - -C "$DEST"
echo "$(find "$DEST" -name '*.py' | wc -l) files in $DEST"
