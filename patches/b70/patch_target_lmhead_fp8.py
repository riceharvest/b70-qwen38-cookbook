"""B70: target LM head in fp8 e4m3 (per-row scale) via torch.ops._xpu_C.fp8_gemm_w8a16.
Halves the 2.5 GB fp16 head read per verify step (4.3 -> 2.2 ms) and frees ~1.3 GB for KV.
Conversion runs at the end of load_weights (so vLLM's weight-memory accounting sees fp8).
When B70_DRAFT_LMHEAD_INT4=1 the draft's int4 head (optionally pruned to B70_DRAFT_VOCAB)
is built from the fp16 weight at the same time, before the fp16 copy is freed.
Run AFTER patch_draft_lmhead_int4.py and patch_draft_vocab.py. Gate: B70_TARGET_LMHEAD_FP8=1.
qsim: KL vs BF16 0.0688 -> 0.0694 on the GPTQ baseline."""
import pathlib, sys
MARK = "# B70_TARGET_FP8"
R = pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm")
M = R / "model_executor/models"

HELPER = '''"""B70 target lm_head fp8 helper (written by patch_target_lmhead_fp8.py)."""
import json, os
import torch


@torch.no_grad()
def convert(model) -> None:
    if os.environ.get("B70_TARGET_LMHEAD_FP8") != "1":
        return
    head = getattr(model, "lm_head", None)
    if head is None or getattr(head, "_b70_fp8", None) is not None:
        return
    W = getattr(head, "weight", None)
    if W is None or W.dim() != 2 or W.numel() == 0 or W.dtype not in (torch.float16, torch.bfloat16):
        return
    if getattr(head, "tp_size", 1) != 1:
        print("[B70] target fp8 head: TP>1, skipped", flush=True); return
    W = W.data
    V, K = W.shape
    if os.environ.get("B70_DRAFT_LMHEAD_INT4") == "1":
        from vllm.model_executor.models.b70_draft_lmhead_int4 import quantize_lmhead_to_int4
        ids, w = None, W
        vf = os.environ.get("B70_DRAFT_VOCAB")
        if vf:
            ids = torch.tensor(sorted(json.load(open(vf))), dtype=torch.long, device=W.device)
            w = W.index_select(0, ids).contiguous()
        head._b70_draft_int4 = (*quantize_lmhead_to_int4(w), ids, V)
        del w
        print(f"[B70] draft int4 head prebuilt ({V if ids is None else ids.numel()} rows)", flush=True)
    Wq = torch.empty((V, K), dtype=torch.float8_e4m3fn, device=W.device)
    sc = torch.empty(V, dtype=torch.float32, device=W.device)
    for i in range(0, V, 8192):
        w = W[i:i + 8192].float()
        s = w.abs().amax(dim=1).clamp(min=1e-8) / 448.0
        sc[i:i + 8192] = s
        Wq[i:i + 8192] = (w / s[:, None]).to(torch.float8_e4m3fn)
    head._b70_fp8 = (Wq.t(), sc)
    head.weight = torch.nn.Parameter(torch.empty((0, K), dtype=W.dtype, device=W.device), requires_grad=False)
    del W
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    print(f"[B70] target lm_head fp8: {V}x{K}, fp16 freed", flush=True)


def logits(head, x: torch.Tensor) -> torch.Tensor:
    Wt, sc = head._b70_fp8
    flat = x.reshape(-1, x.shape[-1])
    out = torch.ops._xpu_C.fp8_gemm_w8a16(flat, Wt, sc, None)
    return out.reshape(*x.shape[:-1], Wt.shape[1])
'''

def sub(p, pairs, count=1):
    s = p.read_text()
    if MARK in s: print("already", p); return
    for old, new in pairs:
        n = s.count(old)
        if n < 1: print("anchor missing", p, old[:70]); sys.exit(1)
        s = s.replace(old, new)
    p.write_text(MARK + "\n" + s); print("patched", p)

(M / "b70_target_lmhead_fp8.py").write_text(HELPER)

# 1) convert after load_weights (inner CausalLM and outer VL wrapper; idempotent)
LW_OLD = ("        loader = AutoWeightsLoader(self)\n"
          "        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n")
LW_NEW = ("        loader = AutoWeightsLoader(self)\n"
          "        _b70_loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
          "        from vllm.model_executor.models.b70_target_lmhead_fp8 import convert as _b70_cv\n"
          "        # inner LM: convert only in the call that loaded lm_head (AutoWeightsLoader may call it once per\n"
          "        # contiguous run of checkpoint keys); outer wrapper: convert at the end (idempotent)\n"
          "        if not hasattr(self, 'lm_head') or any(n.endswith('lm_head.weight') for n in _b70_loaded):\n"
          "            _b70_cv(self if hasattr(self, 'lm_head') else getattr(self, 'language_model', self))\n"
          "        return _b70_loaded\n")
sub(M / "qwen3_5.py", [(LW_OLD, LW_NEW)])

# 2) logits through fp8 when converted
sub(R / "model_executor/layers/logits_processor.py", [(
    '        """Project hidden states through the lm_head, honoring head_dtype."""\n',
    '        """Project hidden states through the lm_head, honoring head_dtype."""\n'
    '        if getattr(lm_head, "_b70_fp8", None) is not None:\n'
    '            from vllm.model_executor.models.b70_target_lmhead_fp8 import logits as _b70_lg\n'
    '            return _b70_lg(lm_head, hidden_states)\n')])

# 3) draft helper: use the prebuilt int4 head (fp16 weight is gone)
sub(M / "b70_draft_lmhead_int4.py", [(
    '    head = getattr(model, "lm_head", None)\n    weight = getattr(head, "weight", None)\n',
    '    head = getattr(model, "lm_head", None)\n    weight = getattr(head, "weight", None)\n'
    '    _pre = getattr(head, "_b70_draft_int4", None)\n'
    '    if _pre is not None:\n'
    '        model._b70_lmhead_int4 = tuple(_pre[:4])\n'
    '        if _pre[4] is not None:\n'
    '            model._b70_draft_vocab_ids = _pre[4]\n'
    '            model._b70_draft_vocab_full = _pre[5]\n'
    '        print("[B70] draft LM head INT4: using prebuilt head", flush=True)\n'
    '        return\n')])
