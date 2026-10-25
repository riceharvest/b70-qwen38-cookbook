#!/usr/bin/env bash
# Rebuild libattn_kernels_xe_2.b70.so (vllm-xpu-kernels release/0.1.15.4 + sycl-tla 87f68506 + the B70 patches: xk-fold, zero-sum-guard, tla-fast3).
# needs: docker, ~25 GB free RAM, Intel oneAPI DPC++ 2026.0 (icpx) unpacked at $ONEAPI.
# usage: ONEAPI=/path/to/oneapi ./build.sh          -> ./libattn_kernels_xe_2.b70.so  (~25 min, single AOT target)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); W=${WORK:-$HERE/work}; mkdir -p "$W"
ONEAPI=${ONEAPI:?set ONEAPI to a oneAPI 2026.0 root containing compiler/2026.0}
IMG=${IMG:-vllm/vllm-openai-xpu:v0.31.0@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7}
JOBS=${JOBS:-2}   # each icpx job peaks ~8 GB; -j16 can take a 64 GB desktop down
[ -d "$W/xk" ]  || { git clone -q -b release/0.1.15.4 https://github.com/vllm-project/vllm-xpu-kernels.git "$W/xk"; git -C "$W/xk" apply "$HERE/xk-fold-b70.patch"; git -C "$W/xk" apply "$HERE/zero-sum-guard.patch"; cp "$HERE"/b70_prod_*.conf "$W/xk/csrc/xpu/attn/kernel_configs/"; }
[ -d "$W/tla" ] || { git clone -q https://github.com/intel/sycl-tla.git "$W/tla"; git -C "$W/tla" checkout -q 87f6850680a580654b9ea2c80dbc01aeb36ad231; git -C "$W/tla" apply "$HERE/tla-fast3-b70.patch"; }
docker run --rm --memory=24g --memory-swap=24g -v "$W:/w" -v "$ONEAPI:/opt/intel/oneapi:ro" \
  -e VLLM_XPU_AOT_DEVICES=bmg-g31-a0 -e VLLM_XPU_XE2_AOT_DEVICES=bmg-g31-a0 --entrypoint bash $IMG -c "
set -eo pipefail
export CMPLR_ROOT=/opt/intel/oneapi/compiler/2026.0 PATH=/opt/intel/oneapi/compiler/2026.0/bin:\$PATH
git config --global --add safe.directory '*'
cd /w/xk
[ -f /w/build/build.ninja ] || cmake -S . -B /w/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=xpu \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.cmake -DVLLM_PYTHON_EXECUTABLE=/opt/venv/bin/python \
  -DVLLM_XPU_ENABLE_XE3P=OFF -DVLLM_XPU_ENABLE_XE_DEFAULT=OFF -DBASIC_KERNELS_ENABLED=OFF -DMOE_KERNELS_ENABLED=OFF \
  -DGDN_KERNELS_ENABLED=OFF -DMQA_LOGITS_KERNELS_ENABLED=OFF -DMHC_KERNELS_ENABLED=OFF -DXPU_SPECIFIC_KERNELS_ENABLED=OFF \
  -DXPUMEM_ALLOCATOR_ENABLED=OFF -DVLLM_XPU_ENABLE_ONEDNN=OFF -DFA2_KERNELS_ENABLED=ON \
  -DVLLM_CHUNK_PREFILL_CONFIG=b70_prod_prefill.conf -DVLLM_PAGED_DECODE_CONFIG=b70_prod_decode.conf -DVLLM_CUTLASS_SRC_DIR=/w/tla
ninja -C /w/build -j $JOBS attn_kernels_xe_2; chown -R $(id -u):$(id -g) /w"
cp "$W/build/libattn_kernels_xe_2.so" "$HERE/libattn_kernels_xe_2.b70.so"; echo "built $HERE/libattn_kernels_xe_2.b70.so"
