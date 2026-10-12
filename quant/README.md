Everything runs inside the pinned vLLM XPU image (it has torch-xpu + transformers). Mount a work dir at `/opt/b70`, the BF16 model at `/bf16`, and an output dir.

```bash
IMG=vllm/vllm-openai-xpu:v0.31.0@sha256:95ac815038c4b3537b173798bfe3b89bdf525a52bd653cd14d4deb01f1062be7
RUN="docker run --rm --memory=24g --device /dev/dri -v /dev/dri:/dev/dri:ro -v $PWD:/opt/b70 -v /path/out:/w --entrypoint python"
# 1. calibration (128 x 2048 tokens) from your own corpora: mixedcal.json ([{text}]), corpus.txt (code), general.txt (prose)
$RUN -v /path/Qwen3.8-27B:/model:ro $IMG -I /opt/b70/build_calib_public.py /opt/b70/calib.json
# 2. quantize (~1 h on a B70, layer-streamed): GPTQ asym int4 g128 + MSE clip search -> raw-zero checkpoint
$RUN -v /path/Qwen3.8-27B:/bf16:ro $IMG -I /opt/b70/b70quant.py /w/q-raw --calib /opt/b70/calib.json
# 3. export to standard GPTQ v1 (qzeros = z-1) and check it with GPTQModel's dequant semantics
$RUN $IMG -I /opt/b70/export_gptq_v1.py /w/q-raw /w/q-gptq && $RUN $IMG -I /opt/b70/check_v1.py /w/q-raw /w/q-gptq
```

`q-raw` needs a `quantize_config.json` before export (see the released model's). Then set the export's `config.json`
`quantization_config` to the GPTQ one (`quant_method: gptq, sym: false, group_size: 128, desc_act: false`, MTP excluded via `dynamic`).

Quality: `bf16_ref.py OUT.pt` stores BF16 top-20 logits for held-out windows (`eval_tokens.json`: `{group: [[ids]]}`).
`REF=OUT.pt qsim.py '{}' LABEL` (BF16 at `/bf16`, quant at `/gptq`) prints KL / top-1 agreement / ppl ratio per group.
`'{"lm_head":"fp8"}'` simulates the fp8 head.
