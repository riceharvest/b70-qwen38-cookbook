"""runtime/spec_packed.py on CPU with a fake vllm_xpu_kernels module: checks the _graph_splits heuristic, the packed
(B, ql, Hkv, G, D) <-> (B, Hkv*ql*G, D) permutation round-trip, fallback conditions, and B70_SPEC_QLEN restore."""
import importlib.util
import os
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "arange"):
    pytest.skip("PyTorch is not installed", allow_module_level=True)

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "runtime" / "spec_packed.py"


@pytest.fixture
def env(monkeypatch):
    calls = {"orig": [], "fa": []}

    def _orig(q, k, v, out, *args, **kwargs):
        calls["orig"].append((q, k, v, out, args, kwargs))
        res = q + 1.0
        if out is not None:
            out.copy_(res)
            return out
        return res

    def flash_attn_varlen_func(qp, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, **kwargs):
        calls["fa"].append({
            "qp": qp.clone(),
            "max_seqlen_q": max_seqlen_q,
            "cu_seqlens_q": cu_seqlens_q.clone(),
            "max_seqlen_k": max_seqlen_k,
            "env_qlen": os.environ.get("B70_SPEC_QLEN"),
            **kwargs,
        })
        # Simple linear transform that preserves shape [B, Hkv*ql*G, D]
        return qp * 2.0

    fai = types.ModuleType("vllm_xpu_kernels.flash_attn_interface")
    fai._spec_decode_varlen_fwd = _orig
    fai.flash_attn_varlen_func = flash_attn_varlen_func
    pkg = types.ModuleType("vllm_xpu_kernels")
    pkg.flash_attn_interface = fai
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", pkg)
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.flash_attn_interface", fai)

    xpu = types.SimpleNamespace(
        capturing=False,
        subslice_count=32,
        is_current_stream_capturing=lambda: xpu.capturing,
        current_device=lambda: 0,
        get_device_properties=lambda dev: types.SimpleNamespace(gpu_subslice_count=xpu.subslice_count),
    )
    monkeypatch.setattr(torch, "xpu", xpu, raising=False)
    monkeypatch.setenv("B70_SPEC_PACKED", "1")

    spec = importlib.util.spec_from_file_location("spec_packed_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return types.SimpleNamespace(mod=mod, fai=fai, calls=calls, xpu=xpu)


def test_graph_splits_heuristic_matches_b70_table(env, monkeypatch):
    gs = env.mod._graph_splits
    q = torch.zeros(1)
    # Not capturing -> None
    env.xpu.capturing = False
    assert gs(q, B=1, Hkv=4, rows=20, D=256) is None

    # Capturing on B70 (32 Xe cores -> target 256 WGs); Qwen3.8 shape: Hkv=4, ql=5, G=4 -> rows=20, D=256 -> tile=8 -> 3 WGs/head -> 12*B WGs
    env.xpu.capturing = True
    assert [gs(q, B=b, Hkv=4, rows=20, D=256) for b in (1, 2, 4, 8)] == [21, 10, 5, 2]
    # Hkv=4, rows=32, D=256 -> tile=8 -> 4 WGs/head -> 16*B WGs -> 256 // (16*B) = 16, 8, 4, 2
    assert [gs(q, B=b, Hkv=4, rows=32, D=256) for b in (1, 2, 4, 8)] == [16, 8, 4, 2]

    # Opt-in 32-row verify tile: rows=20 <= 32, D=256 -> tile=32 -> 1 WG/head -> 4*B WGs -> capped at 32 for B=1,2
    monkeypatch.setenv("B70_SPEC_Q32", "1")
    assert [gs(q, B=b, Hkv=4, rows=20, D=256) for b in (1, 2, 4, 8)] == [32, 32, 16, 8]

    # Disabled via env -> None
    monkeypatch.setenv("B70_GRAPH_KV_SPLITS", "0")
    assert gs(q, B=1, Hkv=4, rows=20, D=256) is None


def test_packed_Verify_permutation_is_exact_roundtrip(env):
    B, ql, Hkv, G, D = 2, 5, 4, 4, 16
    Hq = Hkv * G
    # Give every (b, pos, kv_head, g) a distinct value so any transposed axis breaks the roundtrip check
    q = torch.arange(B * ql * Hq * D, dtype=torch.float32).reshape(B * ql, Hq, D)
    k = torch.zeros(10, 16, Hkv, D)
    v = torch.zeros_like(k)
    out = torch.empty_like(q)
    cu_q = torch.arange(0, (B + 1) * ql, ql, dtype=torch.int32)
    seqused_k = torch.full((B,), 64, dtype=torch.int32)
    block_table = torch.zeros((B, 4), dtype=torch.int32)

    res = env.fai._spec_decode_varlen_fwd(
        q, k, v, out, cu_q, seqused_k, block_table, 64, None, None, 0.125, None, (-1, -1), 0.0
    )
    assert res is out
    assert torch.equal(out, q * 2.0)
    assert len(env.calls["fa"]) == 1 and not env.calls["orig"]
    call = env.calls["fa"][0]
    assert call["qp"].shape == (B, Hkv * ql * G, D)
    assert call["env_qlen"] == "5"
    assert os.environ.get("B70_SPEC_QLEN") == "1"  # restored in finally block
    assert torch.equal(call["cu_seqlens_q"], torch.arange(B + 1, dtype=torch.int32))


def test_packed_falls_back_when_unsupported(env):
    B, ql, Hkv, G, D = 2, 5, 4, 4, 16
    q = torch.ones(B * ql, Hkv * G, D)
    k = torch.zeros(10, 16, Hkv, D)
    cu_q = torch.arange(0, (B + 1) * ql, ql, dtype=torch.int32)
    seq = torch.full((B,), 64, dtype=torch.int32)
    bt = torch.zeros((B, 4), dtype=torch.int32)

    # window_size != (-1, -1) -> fallback
    env.fai._spec_decode_varlen_fwd(q, k, k, None, cu_q, seq, bt, 64, None, None, 0.125, None, (128, 0), 0.0)
    # softcap != 0.0 -> fallback
    env.fai._spec_decode_varlen_fwd(q, k, k, None, cu_q, seq, bt, 64, None, None, 0.125, None, (-1, -1), 30.0)
    # s_aux not None -> fallback
    env.fai._spec_decode_varlen_fwd(q, k, k, None, cu_q, seq, bt, 64, None, None, 0.125, torch.zeros(1), (-1, -1), 0.0)
    # ql == 1 -> fallback
    q1 = torch.ones(B, Hkv * G, D)
    cu1 = torch.arange(B + 1, dtype=torch.int32)
    env.fai._spec_decode_varlen_fwd(q1, k, k, None, cu1, seq, bt, 64, None, None, 0.125, None, (-1, -1), 0.0)

    assert len(env.calls["orig"]) == 4 and not env.calls["fa"]
