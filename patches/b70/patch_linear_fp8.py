"""B70: weight-only fp8 (e4m3, per-output-row scale) for selected UNQUANTIZED linears, via torch.ops._xpu_C.fp8_gemm_w8a16
(same kernel as the fp8 lm_head). For checkpoints that ship some linears in bf16 (e.g. GPTQ MoE quants that keep
attention / linear-attention / shared expert unquantized). Gate: B70_LINEAR_FP8_RE = regex on the layer prefix.
Prefixes containing 'mtp' are never converted. The bf16 copy is freed (default since image b70.4, -2.6 GiB on Qwen3.6-35B-A3B);
B70_LINEAR_FP8_FREE=0 keeps it allocated (the gain is bandwidth either way: decode reads the fp8 copy). Measured on Qwen3.6-35B-A3B (palmfuture GPTQ-Int4): KL vs bf16 0.019."""
import pathlib, sys
MARK = "# B70_LINEAR_FP8"
P = pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/linear.py")
s = P.read_text()
if MARK in s:
    print("already", P); sys.exit(0)
OLD_PW = ("            if (\n                envs.VLLM_XPU_FORCE_N_CONTIG_WEIGHT\n                and weight.ndim == 2\n"
          "                and weight.stride(0) != 1\n            ):\n                layer.weight.data = weight.t().contiguous().t()\n")
NEW_PW = OLD_PW + (
    "            _b70_fp8_convert(layer)\n")
OLD_AP = ("            return linear_batch_invariant(x, layer.weight, bias)\n"
          "        return self._gemm_impl(layer, x, layer.weight, bias)\n")
NEW_AP = ("            return linear_batch_invariant(x, layer.weight, bias)\n"
          "        _f8 = getattr(layer, '_b70_fp8', None)\n"
          "        if _f8 is not None:\n"
          "            _wt, _sc = _f8\n"
          "            _out = torch.ops._xpu_C.fp8_gemm_w8a16(x.reshape(-1, x.shape[-1]).contiguous(), _wt, _sc, None)\n"
          "            _out = _out.reshape(*x.shape[:-1], _wt.shape[1])\n"
          "            return _out + bias if bias is not None else _out\n"
          "        return self._gemm_impl(layer, x, layer.weight, bias)\n")
HELPER = '''
''' + MARK + '''
import os as _b70_os, re as _b70_re
_B70_FP8_RE = _b70_os.environ.get("B70_LINEAR_FP8_RE")
_B70_FP8_N = [0, 0]


@torch.no_grad()
def _b70_fp8_convert(layer) -> None:
    if not _B70_FP8_RE:
        return
    pre = getattr(layer, "prefix", "") or ""
    if "mtp" in pre or not _b70_re.search(_B70_FP8_RE, pre):
        return
    W = layer.weight.data
    if W.ndim != 2 or W.dtype not in (torch.float16, torch.bfloat16) or W.numel() == 0:
        return
    s = W.float().abs().amax(dim=1).clamp(min=1e-8) / 448.0
    Wq = (W.float() / s[:, None]).to(torch.float8_e4m3fn)
    layer._b70_fp8 = (Wq.t(), s.to(torch.float32))
    if _b70_os.environ.get('B70_LINEAR_FP8_FREE', '1') == '1':
        # frees the bf16 copy (~2.6 GiB on Qwen3.6-35B-A3B). On image b70.2 this faulted the GPU under MTP (CAT error);
        # on b70.3 it never reproduced (eager + graph probes found no reader of the freed weight; MTP4 under load with and
        # without prefix caching; 10 min mixed-load stress; same speed as keeping it). B70_LINEAR_FP8_FREE=0 keeps it.
        layer.weight = torch.nn.Parameter(torch.empty((0, W.shape[1]), dtype=W.dtype, device=W.device), requires_grad=False)
    _B70_FP8_N[0] += 1; _B70_FP8_N[1] += W.numel()
    if _B70_FP8_N[0] in (1, 10, 50, 100, 200):
        print(f"[B70] linear fp8: {_B70_FP8_N[0]} layers, {_B70_FP8_N[1] / 1e9:.2f} G params ({pre})", flush=True)
'''
for old, new in ((OLD_PW, NEW_PW), (OLD_AP, NEW_AP)):
    if s.count(old) != 1:
        print("anchor missing/ambiguous", old[:60]); sys.exit(1)
    s = s.replace(old, new)
anchor = "class UnquantizedLinearMethod("
s = s.replace(anchor, HELPER + "\n\n" + anchor, 1)
P.write_text(s); print("patched", P)
