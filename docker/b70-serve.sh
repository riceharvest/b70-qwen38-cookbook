#!/usr/bin/env bash
# Entry point of ghcr.io/riceharvest/b70-vllm. Picks the setup from the model:
#   - a JEV repo (has adapter_vllm/ + serve_decide.py): System 1 /v1/decide + System 2, LoRA, bf16 lm_head
#   - anything else (the Qwen3.8 int4 quants): chat with MTP k=4, fp8 lm_head, tool calls, reasoning parser
# env: MODEL (HF repo id or a path, default darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym), PORT (8000), SERVED_NAME,
#      GPU_UTIL, MAX_MODEL_LEN, FAST (1 = XPU graphs + MTP, 0 = eager), and any extra args after the image name.
set -euo pipefail
MODEL=${MODEL:-darioooooo0o/Qwen3.8-27B-GPTQ-Int4-asym}; PORT=${PORT:-8000}
if [ -d "$MODEL" ]; then DIR=$MODEL; else
  echo "[b70] downloading $MODEL to $HF_HOME (first run only)" >&2
  DIR=$(python -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1]))" "$MODEL"); fi
# B70_MTP_K: MTP draft tokens per step (default 4); capture sizes follow k (batches up to --max-num-seqs 8)
MTP_K=${B70_MTP_K:-4}; case "$MTP_K" in [1-9]) ;; *) echo "B70_MTP_K must be 1-9" >&2; exit 2 ;; esac
CAPTURE=$(python -c "import sys; k=int(sys.argv[1]); print(','.join(map(str, sorted({1, 2, 4, 8} | {(k + 1) * n for n in range(1, 9)}))))" "$MTP_K")
SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K,\"draft_sample_method\":\"probabilistic\"}"
GRAPHS="{\"cudagraph_capture_sizes\":[$CAPTURE]}"
if [ "${FAST:-1}" = 1 ]; then FASTARGS=(--speculative-config "$SPEC" --compilation-config "$GRAPHS")
else export VLLM_XPU_ENABLE_XPU_GRAPH=0; FASTARGS=(--enforce-eager); fi
COMMON=(--port "$PORT" --distributed-executor-backend mp --quantization gptq --dtype float16 --kv-cache-dtype fp8
        --max-num-seqs 8 --max-num-batched-tokens 8192 --cpu-offload-gb 4 --cpu-offload-params embed_tokens visual)
if [ -d "$DIR/adapter_vllm" ] && [ -f "$DIR/serve_decide.py" ]; then
  # LoRA buffers follow the lm_head dtype, so the fp8 head stays off. --max-num-seqs 8 is required by JEV.
  export B70_TARGET_LMHEAD_FP8=0 JEV_DECIDE_CALIBRATION="$DIR/calibration.json"
  exec python "$DIR/serve_decide.py" --model "$DIR" --served-model-name "${SERVED_NAME:-autotrust/JEV-27B-VL}" "${COMMON[@]}" \
    --enable-lora --max-lora-rank 32 --max-loras 1 --lora-modules jev-decision="$DIR/adapter_vllm" \
    --logprobs-mode processed_logprobs --max-logprobs 64 --enable-prefix-caching --mamba-cache-mode align \
    --limit-mm-per-prompt '{"image": 8}' --trust-request-chat-template \
    --max-model-len "${MAX_MODEL_LEN:-131072}" --gpu-memory-utilization "${GPU_UTIL:-0.90}" "${FASTARGS[@]}" "$@"
fi
export B70_TARGET_LMHEAD_FP8=${B70_TARGET_LMHEAD_FP8:-1}
# linears a checkpoint ships in bf16 (e.g. attention/GDN/shared expert in GPTQ MoE quants) -> fp8 weight-only
[ "${B70_LINEAR_FP8:-1}" = 1 ] && export B70_LINEAR_FP8_RE="${B70_LINEAR_FP8_RE-(self_attn\.(qkv_proj|o_proj)|linear_attn\.(in_proj_qkvz|in_proj_qkv|in_proj_z|out_proj)|shared_expert\.(gate_up_proj|down_proj))$}"
exec vllm serve "$DIR" --served-model-name "${SERVED_NAME:-qwen38}" "${COMMON[@]}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" --gpu-memory-utilization "${GPU_UTIL:-0.93}" "${FASTARGS[@]}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --generation-config auto "$@"
