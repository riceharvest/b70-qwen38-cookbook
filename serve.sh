#!/usr/bin/env bash
# Qwen3.8-27B on one Intel Arc Pro B70 (32 GB): asym GPTQ int4 + MTP k=4 + fp8 KV + 262K context + vision.
# usage: MODEL_DIR=/path/to/downloaded/model ./serve.sh        (env: PORT, NAME, GPU_UTIL, SO_FILE, B70_SPEC_PACKED)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the downloaded model directory}
PORT=${PORT:-8000}; NAME=${NAME:-b70-qwen38}; GPU_UTIL=${GPU_UTIL:-0.93}
IMG=vllm/vllm-openai-xpu:v0.31.0@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7
SO=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels_xe_2.so
SO_FILE=${SO_FILE:-$HERE/kernels/libattn_kernels_xe_2.b70.so}
# B70_SPEC_PACKED=1 (packed MTP verify) corrupts some outputs ("!!!!") with ~8 long generations in flight, so it's off
KMOUNT=(); SPEC_PACKED=${B70_SPEC_PACKED:-0}
if [ -f "$SO_FILE" ]; then
  KMOUNT=(-v "$SO_FILE:$SO:ro")   # built for THIS image only (kernels/README.md)
else
  echo "note: $SO_FILE not found -> stock attention kernels, packed MTP verify off (slower; see kernels/README.md)" >&2
  SPEC_PACKED=0
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:$PORT" --device /dev/dri -v /dev/dri:/dev/dri:ro --shm-size 8g \
  -v "$MODEL_DIR:/model:ro" -v "$HERE/patches:/patches:ro" -v "$HERE/runtime:/opt/b70:ro" "${KMOUNT[@]}" \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0} -e CCL_ATL_TRANSPORT=mpi -e CCL_PROCESS_LAUNCHER=none \
  -e B70_MTP_BF16_DRAFT=1 -e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e B70_DRAFT_VOCAB=/opt/b70/draft_vocab_64k.json -e B70_TARGET_LMHEAD_FP8=${B70_TARGET_LMHEAD_FP8:-1} -e B70_SPEC_PACKED=$SPEC_PACKED \
  -e PORT=$PORT -e GPU_UTIL=$GPU_UTIL \
  --entrypoint bash $IMG -lc '
set -e
for p in patch_mtp_nightly patch_mtp_boundary patch_draft_lmhead_int4 patch_draft_mtp_int4 patch_gdn_mixed_split_v5 \
         patch_fix_accepted_sync patch_fix_backward_copy patch_fix_eagle_drop; do python /patches/sergiiob/$p.py; done
for p in patch_embed_uva patch_draft_vocab patch_asym_gptq patch_target_lmhead_fp8 patch_spec_packed; do python /patches/b70/$p.py; done
exec vllm serve /model --port $PORT --served-model-name qwen38 --distributed-executor-backend mp \
  --quantization gptq --dtype float16 --kv-cache-dtype fp8 --max-model-len 262144 --gpu-memory-utilization $GPU_UTIL \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4,\"draft_sample_method\":\"probabilistic\"}" \
  --compilation-config "{\"cudagraph_capture_sizes\":[1,2,4,5,8,10,15,20,25,30,35,40]}" \
  --cpu-offload-gb 4 --cpu-offload-params embed_tokens visual \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --generation-config auto'
echo "starting $NAME on :$PORT (first start ~2-3 min; logs: docker logs -f $NAME)"
for i in $(seq 1 200); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "$NAME READY on :$PORT"; exit 0; }; sleep 3; done
echo "not ready yet - docker logs $NAME" >&2; exit 1
