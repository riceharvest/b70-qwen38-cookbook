#!/usr/bin/env bash
# Copy the .py files of vllm + vllm_xpu_kernels out of the pinned base image (the Dockerfile's FROM), for
# tests/test_patches.py. No GPU, no network inside the container. ~40 MB.
# The default cache dir is keyed on the FROM reference, so bumping the base image fetches fresh sources.
# usage: tests/fetch_vllm_src.sh [--if-missing] [dest]
set -euo pipefail
cd "$(dirname "$0")/.."
IF_MISSING=0; [ "${1:-}" = --if-missing ] && { IF_MISSING=1; shift; }
IMG=$(sed -n 's/^FROM \([^ ]*\).*/\1/p' docker/Dockerfile | head -1)
KEY=$(printf %s "$IMG" | sha256sum | cut -c1-12)
DEST=${1:-${B70_VLLM_SRC:-$HOME/.cache/b70-cookbook-tests/$KEY/site-packages}}
if [ $IF_MISSING = 1 ] && [ -f "$DEST/vllm/__init__.py" ]; then exit 0; fi
TMP="$DEST.partial"; rm -rf "$TMP"; mkdir -p "$TMP"
docker run --rm --network none --entrypoint bash "$IMG" -c \
  "cd /opt/venv/lib/python3.12/site-packages && find vllm vllm_xpu_kernels -name '*.py' | tar -cf - -T -" | tar -xf - -C "$TMP"
rm -rf "$DEST"; mv "$TMP" "$DEST"
echo "$(find "$DEST" -name '*.py' | wc -l) files from $IMG in $DEST"
