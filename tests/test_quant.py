"""The quantizer's numerics on CPU (needs torch + safetensors; skipped without them).

quant/b70quant.py runs a whole model at import time, so the functions are pulled out of its source with `ast` and run
unchanged on small tensors. What is checked: the MSE clip search, the packed layout the loaders expect (qweight
LSB-first along K, qzeros along N), GPTQ's error compensation, and the GPTQ v1 export + its independent check."""
import ast
import json
import pathlib
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "arange"):  # an empty `torch` namespace directory on sys.path is not PyTorch
    pytest.skip("PyTorch is not installed", allow_module_level=True)
safetensors_torch = pytest.importorskip("safetensors.torch")

QUANT = pathlib.Path(__file__).resolve().parent.parent / "quant"
G = 16  # small group size so the tensors stay tiny; the code takes G as a global


@pytest.fixture(scope="module")
def q():
    src = (QUANT / "b70quant.py").read_text()
    want = {"GRID", "find_params", "gptq", "dequant", "pack"}
    keep = []
    for node in ast.parse(src).body:
        names = {t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)}
        if (isinstance(node, ast.FunctionDef) and node.name in want) or (names & want):
            keep.append(node)
    assert {getattr(n, "name", None) for n in keep if isinstance(n, ast.FunctionDef)} == want - {"GRID"}
    ns = {"torch": torch, "G": G, "dev": "cpu", "a": types.SimpleNamespace(damp=0.01)}
    exec(compile(ast.Module(keep, []), "b70quant.py", "exec"), ns)  # noqa: S102
    return types.SimpleNamespace(**ns)


def unpack_like_a_loader(qw, qz, sc):
    """Dequantize the way vLLM / GPTQModel read the tensors (raw zero points, as written by b70quant)."""
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    w = ((qw.unsqueeze(1) >> sh.view(1, 8, 1)) & 0xF).reshape(-1, qw.shape[1]).float()        # [K, N]
    z = ((qz.unsqueeze(-1) >> sh.view(1, 1, 8)) & 0xF).reshape(qz.shape[0], -1).float()       # [K/G, N]
    g = torch.arange(w.shape[0]) // G
    return (w - z[g]) * sc.float()[g]


def test_find_params_beats_plain_minmax(q):
    torch.manual_seed(0)
    w = torch.randn(64, G)
    w[:, 0] *= 8  # an outlier per group: clipping a little should win
    s, z = q.find_params(w)
    err = lambda s, z: (((torch.clamp(torch.round(w / s) + z, 0, 15) - z) * s - w) ** 2).sum()
    s1 = ((w.amax(1, keepdim=True).clamp(min=0) - w.amin(1, keepdim=True).clamp(max=0)) / 15).clamp(min=1e-8)
    z1 = torch.round(-w.amin(1, keepdim=True).clamp(max=0) / s1)
    assert err(s, z) < err(s1, z1)
    assert ((z >= 0) & (z <= 15)).all() and (z == z.round()).all()  # zero points are integers in the 4-bit range


def test_find_params_zero_group_is_safe(q):
    s, z = q.find_params(torch.zeros(4, G))
    assert torch.isfinite(s).all() and torch.isfinite(z).all() and (s > 0).all()


def test_pack_layout_roundtrip(q):
    torch.manual_seed(1)
    N, K = 24, 64
    Q = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    S = torch.rand(N, K // G) + 0.1
    Z = torch.randint(0, 16, (N, K // G)).float()
    qw, sc, qz = q.pack(Q, S, Z)
    assert (qw.dtype, qw.shape) == (torch.int32, (K // 8, N))          # packed along K
    assert (qz.dtype, qz.shape) == (torch.int32, (K // G, N // 8))      # zeros packed along N
    assert (sc.dtype, sc.shape) == (torch.float16, (K // G, N))
    # a loader that reads the layout recovers exactly what dequant() says
    assert torch.allclose(unpack_like_a_loader(qw, qz, sc), q.dequant(Q, S.half().float(), Z).t(), atol=1e-6)


def test_pack_is_lsb_first(q):
    Q = torch.zeros(8, G, dtype=torch.uint8)
    Q[0, :8] = torch.arange(8)  # output channel 0, k = 0..7 -> nibbles 0..7 of qweight[0, 0]
    qw, _, _ = q.pack(Q, torch.ones(8, 1), torch.zeros(8, 1))
    assert int(qw[0, 0]) == 0x76543210
    assert int(qw[0, 1]) == 0  # the next output channel is a different column


def test_dequant_error_within_half_a_step(q):
    torch.manual_seed(2)
    W = torch.randn(16, 64)
    Hs = torch.eye(64)  # identity Hessian: GPTQ degenerates to per-weight rounding
    Q, S, Z = q.gptq(W, Hs)
    assert Q.dtype == torch.uint8 and int(Q.max()) <= 15
    err = (q.dequant(Q, S, Z) - W).abs()
    step = S.repeat_interleave(G, 1)
    # with clip < 1 a few weights saturate; everything else is within half a quantization step
    assert (err <= step / 2 + 1e-5).float().mean() > 0.95


def test_gptq_beats_rounding_on_correlated_inputs(q):
    torch.manual_seed(3)
    N, K, T = 32, 64, 512
    base = torch.randn(T, 8)
    X = base @ torch.randn(8, K) + 0.05 * torch.randn(T, K)  # strongly correlated features
    W = torch.randn(N, K)
    H = 2 * X.t() @ X / T
    Qg, Sg, Zg = q.gptq(W, H)
    # plain round-to-nearest with the same per-group parameters
    Qr = torch.zeros_like(Qg)
    Wr = torch.zeros_like(W)
    for g in range(K // G):
        s, z = q.find_params(W[:, g * G:(g + 1) * G])
        qq = torch.clamp(torch.round(W[:, g * G:(g + 1) * G] / s) + z, 0, 15)
        Wr[:, g * G:(g + 1) * G] = (qq - z) * s
    out_err = lambda Wq: ((X @ (Wq - W).t()) ** 2).mean()
    assert out_err(q.dequant(Qg, Sg, Zg)) < out_err(Wr)  # error compensation pays off on the layer output


def test_gptq_handles_dead_inputs_and_singular_hessians(q):
    torch.manual_seed(4)
    W = torch.randn(8, 32)
    H = torch.zeros(32, 32)
    H[:16, :16] = torch.eye(16)  # the last 16 input channels never fire
    Q, S, Z = q.gptq(W, H)
    assert torch.isfinite(q.dequant(Q, S, Z)).all()
    assert (q.dequant(Q, S, Z)[:, 16:].abs() < 1e-5).all()  # dead columns are zeroed, and the block-diagonal Hessian leaks nothing into them


def test_export_v1_and_independent_check(q, tmp_path):
    """Raw-zero checkpoint -> GPTQ v1 (qzeros store z-1; z=0 groups get z=1 and q+1) -> check_v1 says nothing differs."""
    torch.manual_seed(5)
    N, K = 16, 64
    Q = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    S = torch.rand(N, K // G) + 0.1
    Z = torch.randint(1, 16, (N, K // G)).float()
    Z[0, 0] = 0  # a group whose zero point v1 cannot store
    Q[0, :G] = torch.randint(0, 14, (G,), dtype=torch.uint8)  # leaves headroom for the q+1 shift
    qw, sc, qz = q.pack(Q, S, Z)
    raw, v1 = tmp_path / "raw", tmp_path / "v1"
    raw.mkdir()
    safetensors_torch.save_file({"m.qweight": qw, "m.scales": sc, "m.qzeros": qz, "norm.weight": torch.ones(4)},
                                str(raw / "model-layer00.safetensors"))
    (raw / "quantize_config.json").write_text(json.dumps({"bits": 4, "group_size": G, "sym": False}))
    (raw / "config.json").write_text("{}")
    r = subprocess.run([sys.executable, str(QUANT / "export_gptq_v1.py"), str(raw), str(v1)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "z=0 groups fixed: 1" in r.stdout
    out = json.loads((v1 / "quantize_config.json").read_text())
    assert out["checkpoint_format"] == "gptq" and out["meta"]["zero_format"] == "gptq v1 (stored z-1)"
    assert (v1 / "config.json").exists()  # non-weight files are carried over
    r = subprocess.run([sys.executable, str(QUANT / "check_v1.py"), str(raw), str(v1)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith(f"0 of {N * K} weights differ"), r.stdout
