#!/usr/bin/env bash
# head-128 kernel build for Qwen3.6-35B-A3B-TinyKV: oneAPI 2026.0 from the Intel DLE image if not given, then build.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ONEAPI=${ONEAPI:-/mnt/ssd/b70-prep/oneapi-2026}
if [ ! -x "$ONEAPI/compiler/2026.0/bin/icpx" ]; then
  docker pull -q intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
  c=$(docker create intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04); mkdir -p "$ONEAPI"
  docker cp "$c:/opt/intel/oneapi/compiler" "$ONEAPI/"; docker rm "$c" >/dev/null
fi
ONEAPI=$ONEAPI JOBS=${JOBS:-2} WORK=${WORK:-/mnt/ssd/b70-code/cb-tinykv-work} "$HERE/build.sh"
