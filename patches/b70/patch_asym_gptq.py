"""B70: load asymmetric GPTQ (sym=false, uint4 + qzeros) on XPU.
- auto_gptq: accept (4, False) -> uint4 and pass zero_points=True to the kernel.
- Checkpoints are standard GPTQ v1 (qzeros store z-1, export_gptq_v1.py); on XPU add the 1 back
  at load (qzeros += 0x11111111), since int4_gemm_w4a16 takes raw zero points.
- XPUwNa16: keep qzeros as stored ([K/G, N/8], packed along N); that is the layout
  int4_gemm_w4a16 expects (verified in zp_check.py). Stock code transposes it, which is wrong."""
import pathlib, sys
MARK = "# B70_ASYM_GPTQ"
R = pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm")

def sub(p, pairs):
    s = p.read_text()
    if MARK in s: return
    for old, new in pairs:
        if old not in s: print("anchor missing", p, old[:60]); sys.exit(1)
        s = s.replace(old, new, 1)
    p.write_text(MARK + "\n" + s); print("patched", p)

sub(R / "model_executor/layers/quantization/auto_gptq.py", [
    ("        (4, True): scalar_types.uint4b8,\n",
     "        (4, True): scalar_types.uint4b8,\n        (4, False): scalar_types.uint4,\n"),
    ("        verify_marlin_supported(\n            quant_type=self.quant_config.quant_type,\n",
     "        if self.quant_config.is_sym: verify_marlin_supported(\n            quant_type=self.quant_config.quant_type,\n"),
    ("            group_size=self.quant_config.group_size,\n            zero_points=False,\n",
     "            group_size=self.quant_config.group_size,\n            zero_points=not self.quant_config.is_sym,\n"),
    ("    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
     "        self.kernel.process_weights_after_loading(layer)\n",
     "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
     "        from vllm.platforms import current_platform\n"
     "        if not self.quant_config.is_sym and current_platform.is_xpu():\n"
     "            layer.qzeros.data += 0x11111111  # GPTQ v1 stores z-1; XPU kernel takes raw z\n"
     "        self.kernel.process_weights_after_loading(layer)\n"),
])
sub(R / "model_executor/kernels/linear/mixed_precision/xpu.py", [
    ("            getattr(layer, self.w_zp_name).data = (\n"
     "                getattr(layer, self.w_zp_name).data.t().contiguous()\n            )\n",
     "            pass  # B70: qzeros already [K/G, N/8] packed along N\n"),
])
