"""B70: route decode-sized int4 MoE grouped GEMMs (<= B70_MOE_GEMV_MAXROWS rows, default 8) to b70_moe::gemv
(kernels/moe-gemv/moe_int4_gemv.sycl), which replaces the CUTLASS grouped GEMM's ~40 us fixed cost per call at decode.
Needs the extension at $B70_MOE_GEMV_SO (default /opt/b70moe/b70_moe_gemv.so). Gate: B70_MOE_GEMV=1.
Debug: B70_MOE_GEMV_CHECK=1 (eager only) also runs CUTLASS and dumps inputs on a >1% row mismatch to $B70_MOE_GEMV_CHECK_DIR;
B70_MOE_GEMV_ONLY_N=<N> limits the GEMV to one projection width."""
import pathlib, sys
MARK = "# B70_MOE_GEMV"
P = pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/fused_moe_interface.py")
s = P.read_text()
if MARK in s:
    print("already", P); sys.exit(0)
HELPER = MARK + '''
import os as _b70os
_B70_GEMV = _b70os.environ.get("B70_MOE_GEMV") == "1"
_B70_MAXROWS = int(_b70os.environ.get("B70_MOE_GEMV_MAXROWS", "8"))
if _B70_GEMV:
    torch.ops.load_library(_b70os.environ.get("B70_MOE_GEMV_SO", "/opt/b70moe/b70_moe_gemv.so"))
    try:
        @torch.library.register_fake("b70_moe::gemv")
        def _b70_gemv_fake(A, B, S, rows, D, N, K, E, G, nr, low_first):
            return None
    except Exception:
        pass


def _b70_grouped_gemm(is_int4, **kw):
    A, B, S, D = kw["ptr_A"], kw["ptr_B"], kw.get("ptr_B_scale"), kw["ptr_D"]
    K, N, E = kw["K"], kw["N"], kw["num_experts"]
    if (_B70_GEMV and is_int4 and kw.get("ptr_A_scale") is None and kw.get("ptr_bias") is None and S is not None
            and S.dim() == 3 and A.dtype in (torch.float16, torch.bfloat16) and S.dtype == A.dtype
            and A.shape[0] <= _B70_MAXROWS and K % 512 == 0
            and (_b70os.environ.get("B70_MOE_GEMV_ONLY_N") in (None, "", str(N))) and K % S.shape[-1] == 0 and (K // S.shape[-1]) % 32 == 0
            and A.is_contiguous() and B.is_contiguous() and S.is_contiguous() and D.is_contiguous()):
        torch.ops.b70_moe.gemv(A, B, S, kw["rows_per_expert"], D, N, K, E, K // S.shape[-1], 4, True)
        if _b70os.environ.get("B70_MOE_GEMV_CHECK") == "1":   # debug: compare with CUTLASS, dump on mismatch (eager only)
            ref = torch.empty_like(D); kw2 = dict(kw); kw2["ptr_D"] = ref
            torch.ops._xpu_C.cutlass_grouped_gemm_interface(**kw2)
            rel = ((D.float() - ref.float()).norm(dim=1) / ref.float().norm(dim=1).clamp(min=1e-6))
            global _B70_NCHK
            _B70_NCHK = globals().get("_B70_NCHK", 0) + 1
            if bool(torch.isnan(D).any()) or float(rel.max()) > 1e-2:
                rows = kw["rows_per_expert"]; ex = torch.nonzero(rows).flatten()
                fn = _b70os.path.join(_b70os.environ.get("B70_MOE_GEMV_CHECK_DIR", "/tmp"), f"b70_gemv_mismatch_{_B70_NCHK}.pt")
                torch.save({"A": A.cpu(), "rows": rows.cpu(), "experts": ex.cpu(), "B": B[ex].cpu(), "S": S[ex].cpu(), "D": D.cpu(), "ref": ref.cpu(),
                            "N": N, "K": K, "E": E, "rel": rel.cpu()}, fn)
                print(f"[B70] GEMV MISMATCH call {_B70_NCHK} N={N} K={K} max_row_rel={float(rel.max()):.3g} nan={bool(torch.isnan(D).any())} -> {fn}", flush=True)
        return
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(**kw)

'''
anchor = "class XpuFusedMoe:"
assert s.count(anchor) == 1
s = s.replace(anchor, HELPER + "\n" + anchor, 1)
i = s.index("    def _apply_kernel(")
body = s[i:]
n = body.count("        torch.ops._xpu_C.cutlass_grouped_gemm_interface(\n")
assert n == 2, f"expected 2 grouped GEMM calls in _apply_kernel, found {n}"
body = body.replace("        torch.ops._xpu_C.cutlass_grouped_gemm_interface(\n", "        _b70_grouped_gemm(self.is_int4,\n")
P.write_text(s[:i] + body); print("patched", P)
