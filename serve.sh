#!/usr/bin/env bash
# Qwen3.8-27B on one Intel Arc Pro B70 (32 GB): asym GPTQ int4 + MTP k=4 + fp8 KV + 262K context + vision.
# usage: MODEL_DIR=/path/to/downloaded/model ./serve.sh        (env: PORT, NAME, GPU_UTIL, SO_FILE, B70_SPEC_PACKED, B70_GDN_STATE, B70_MTP_K)
# B70_MTP_K: MTP draft tokens per step (default 4). Code/JSON/reasoning accept deep drafts often enough that 5 measured
#   +0.4-0.6 tokens/step (prose: no gain) for one more draft pass (~0.7 ms); graph capture sizes follow k.
# B70_PREFIX_UNIT=128: finer prefix-cache hits (--prefix-match-unit; must divide 1664). Multi-turn / tool-loop requests
#   then re-prefill ~200 instead of ~1.7-3.3K tokens of history. Opt-in until validated (MTP + prefix cache class #53912).
# B70_GDN_STATE=fp16: store the linear-attention (GDN) state in fp16 instead of the model config's fp32 (halves it; for many
#   concurrent requests, see README "Many users at once")
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/docker/overlay.sh" "$HERE/build/overlay" >/dev/null   # the B70 fork changes (docker/fork.env), copied over the stock image below
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the downloaded model directory}
PORT=${PORT:-8000}; NAME=${NAME:-b70-qwen38}; GPU_UTIL=${GPU_UTIL:-0.93}
IMG=vllm/vllm-openai-xpu:v0.32.0@sha256:20509fabe50dda62c09c0786d1a6effa299d3dbdddcc49331b93e65144652a8c
SO=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels_xe_2.so
SO_FILE=${SO_FILE:-$HERE/kernels/libattn_kernels_xe_2.b70.so}
# packed MTP verify needs the v2 kernel (zero-sum guard); the v1 .so corrupts some outputs ("!!!!") with it on
KMOUNT=(); SPEC_PACKED=${B70_SPEC_PACKED:-1}
# MoE models: int4 decode GEMV (kernels/b70_moe_gemv.so, release v3) + fp8 weight-only for linears a checkpoint ships in bf16
MOE_SO=${MOE_SO:-$HERE/kernels/b70_moe_gemv.so}; MOE_GEMV=0; MOEMOUNT=()
[ -f "$MOE_SO" ] && { MOEMOUNT=(-v "$MOE_SO:/opt/b70moe/b70_moe_gemv.so:ro"); MOE_GEMV=${B70_MOE_GEMV:-1}; }
GDN_ARGS=""; case "${B70_GDN_STATE:-}" in fp16) GDN_ARGS="--mamba-ssm-cache-dtype float16" ;; bf16) GDN_ARGS="--mamba-ssm-cache-dtype bfloat16" ;; ""|fp32) ;; *) echo "B70_GDN_STATE must be fp16, bf16 or fp32" >&2; exit 2 ;; esac
LINEAR_FP8_RE=${B70_LINEAR_FP8_RE-'(self_attn\.(qkv_proj|o_proj)|linear_attn\.(in_proj_qkvz|in_proj_qkv|in_proj_z|out_proj)|shared_expert\.(gate_up_proj|down_proj))$'}; [ "${B70_LINEAR_FP8:-1}" = 0 ] && LINEAR_FP8_RE=""
if [ -f "$SO_FILE" ]; then
  KMOUNT=(-v "$SO_FILE:$SO:ro")   # built for THIS image only (kernels/README.md)
  if [ "$SPEC_PACKED" = 1 ] && sha256sum "$SO_FILE" | grep -q '^aa241d52a601'; then
    echo "note: $SO_FILE is the v1 kernel (no zero-sum guard) -> packed MTP verify off; get v2: gh release download v2" >&2
    SPEC_PACKED=0
  fi
else
  echo "note: $SO_FILE not found -> stock attention kernels, packed MTP verify off (slower; see kernels/README.md)" >&2
  SPEC_PACKED=0
fi
PREFIX_ARGS=""; [ -n "${B70_PREFIX_UNIT:-}" ] && PREFIX_ARGS="--prefix-match-unit ${B70_PREFIX_UNIT}"
MTP_K=${B70_MTP_K:-4}; case "$MTP_K" in [1-9]) ;; *) echo "B70_MTP_K must be 1-9" >&2; exit 2 ;; esac
CAPTURE=$(python3 -c "import sys; k=int(sys.argv[1]); print(','.join(map(str, sorted({1, 2, 4, 8} | {(k + 1) * n for n in range(1, 9)}))))" "$MTP_K")
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:$PORT" --device /dev/dri -v /dev/dri:/dev/dri:ro --shm-size 8g \
  -v "$MODEL_DIR:/model:ro" -v "$HERE/build/overlay:/b70/overlay:ro" -v "$HERE/runtime:/opt/b70:ro" "${KMOUNT[@]}" "${MOEMOUNT[@]}" \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0} -e CCL_ATL_TRANSPORT=mpi -e CCL_PROCESS_LAUNCHER=none \
  -e B70_MTP_BF16_DRAFT=1 -e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e B70_DRAFT_VOCAB=/opt/b70/draft_vocab_64k.json -e B70_TARGET_LMHEAD_FP8=${B70_TARGET_LMHEAD_FP8:-1} -e B70_SPEC_PACKED=$SPEC_PACKED \
  -e B70_MOE_GEMV=$MOE_GEMV -e "B70_LINEAR_FP8_RE=$LINEAR_FP8_RE" \
  -e PORT=$PORT -e GPU_UTIL=$GPU_UTIL -e "B70_GDN_ARGS=$GDN_ARGS" -e MTP_K=$MTP_K -e CAPTURE=$CAPTURE -e "PREFIX_ARGS=$PREFIX_ARGS" \
  --entrypoint bash $IMG -lc '
set -e
SP=/opt/venv/lib/python3.12/site-packages; cp -r /b70/overlay/site-packages/. $SP/; (cd $SP && xargs -r rm -- < /b70/overlay/removed.txt)
CT=$(python /opt/b70/inline_system_template.py /model /tmp/chat_template_b70.jinja)  # Claude Code /v1/messages: keep the prefix cache
exec vllm serve /model ${CT:+--chat-template $CT} --port $PORT --served-model-name qwen38 --distributed-executor-backend mp \
  --quantization gptq --dtype float16 --kv-cache-dtype fp8 --max-model-len 262144 --gpu-memory-utilization $GPU_UTIL \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K,\"draft_sample_method\":\"probabilistic\"}" \
  --compilation-config "{\"cudagraph_capture_sizes\":[$CAPTURE]}" \
  --cpu-offload-gb 4 --cpu-offload-params embed_tokens visual \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --generation-config auto --enable-prompt-tokens-details $B70_GDN_ARGS $PREFIX_ARGS'
echo "starting $NAME on :$PORT (first start ~2-3 min; logs: docker logs -f $NAME)"
for _ in $(seq 1 200); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "$NAME READY on :$PORT"; exit 0; }; sleep 3; done
echo "not ready yet - docker logs $NAME" >&2; exit 1
