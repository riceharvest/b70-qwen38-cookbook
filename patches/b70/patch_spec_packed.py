"""B70: load spec_packed (single-pass packed MTP-verify attention) in every process that imports
vllm_xpu_kernels.flash_attn_interface. Needs the spec-mask libattn_kernels_xe_2 (.spec.so). Gate: B70_SPEC_PACKED=1."""
import pathlib
MARK = "# B70_SPEC_PACKED"
p = pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/flash_attn_interface.py")
s = p.read_text()
if MARK not in s:
    s += (f"\n{MARK}\nimport os as _b70os\nif _b70os.environ.get('B70_SPEC_PACKED') == '1':\n"
          "    import sys as _b70sys\n    _b70sys.path.insert(0, '/opt/b70')\n    import spec_packed  # noqa: F401\n")
    p.write_text(s); print("patched", p)
