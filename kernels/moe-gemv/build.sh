#!/usr/bin/env bash
# Build kernels/b70_moe_gemv.so (int4 MoE decode GEMV) inside the pinned vLLM XPU image. CPU only, ~1 min.
# Needs icpx 2026.0 (oneAPI base toolkit): ONEAPI=/path/to/oneapi ./build.sh   -> ../b70_moe_gemv.so
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ONEAPI=${ONEAPI:?set ONEAPI to your oneAPI install (with compiler/2026.0)}
IMG=vllm/vllm-openai-xpu:v0.31.0@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7
W=$(mktemp -d); cp "$HERE/moe_int4_gemv.sycl" "$W/"
docker run --rm --memory=12g -v "$W:/g" -v "$ONEAPI:/opt/intel/oneapi:ro" -e TORCH_XPU_ARCH_LIST=bmg-g31 -e MAX_JOBS=2 --entrypoint bash $IMG -lc '
  export PATH=/opt/intel/oneapi/compiler/2026.0/bin:$PATH   # compiler only: keep torch'"'"'s own SYCL runtime libs
  python -c "import torch.utils.cpp_extension as ce; ce.load(name=\"b70_moe_gemv\", sources=[\"/g/moe_int4_gemv.sycl\"], build_directory=\"/g\", is_python_module=False, extra_cflags=[\"-O3\"], extra_sycl_cflags=[\"-O3\"])"
  chown -R '"$(id -u):$(id -g)"' /g'
cp "$W/b70_moe_gemv.so" "$HERE/../b70_moe_gemv.so"; rm -rf "$W"; echo "built $HERE/../b70_moe_gemv.so"
