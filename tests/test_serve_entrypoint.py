"""docker/b70-serve.sh (the image entrypoint) on CPU: run it with stub `vllm` and `python` so we can check which
arguments and environment it would hand to vLLM, without a GPU, a model download or the image."""
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "docker" / "b70-serve.sh"

# `python -c ...` runs for real (the script computes the graph capture sizes that way). The two scripts it launches
# are recorded instead: inline_system_template.py prints nothing (= no custom chat template), serve_decide.py and
# `vllm serve` dump their argv and the B70_* / JEV_* environment to $B70_STUB_OUT.
PYTHON_STUB = f"""#!/usr/bin/env bash
if [ "$1" = "-c" ]; then exec {sys.executable} "$@"; fi
case "$1" in
  */inline_system_template.py) exit 0 ;;
  *) {sys.executable} - "$@" <<'PY'
import json, os, sys
env = {{k: v for k, v in os.environ.items() if k.startswith(("B70_", "JEV_", "VLLM_"))}}
json.dump({{"cmd": "serve_decide", "argv": sys.argv[1:], "env": env}}, open(os.environ["B70_STUB_OUT"], "w"))
PY
  ;;
esac
"""
VLLM_STUB = f"""#!/usr/bin/env bash
{sys.executable} - "$@" <<'PY'
import json, os, sys
env = {{k: v for k, v in os.environ.items() if k.startswith(("B70_", "JEV_", "VLLM_"))}}
json.dump({{"cmd": "vllm", "argv": sys.argv[1:], "env": env}}, open(os.environ["B70_STUB_OUT"], "w"))
PY
"""


@pytest.fixture
def run(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("python", PYTHON_STUB), ("vllm", VLLM_STUB)):
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()

    def _run(env=None, args=(), jev=False):
        if jev:
            (model / "adapter_vllm").mkdir(exist_ok=True)
            (model / "serve_decide.py").write_text("")
        out = tmp_path / "out.json"
        full_env = {"PATH": f"{bindir}:{os.environ['PATH']}", "HOME": str(tmp_path), "HF_HOME": str(tmp_path),
                    "MODEL": str(model), "B70_STUB_OUT": str(out)}
        full_env.update(env or {})
        r = subprocess.run(["bash", str(SCRIPT), *args], env=full_env, capture_output=True, text=True, check=False)
        return r, (json.loads(out.read_text()) if out.exists() else None)

    return _run


def arg(argv, flag):
    return argv[argv.index(flag) + 1]


def test_default_chat_setup(run):
    r, out = run()
    assert r.returncode == 0, r.stderr
    argv, env = out["argv"], out["env"]
    assert out["cmd"] == "vllm" and argv[:2] == ["serve", str(pathlib.Path(argv[1]))]
    assert arg(argv, "--served-model-name") == "qwen38"
    assert arg(argv, "--port") == "8000"
    assert arg(argv, "--max-model-len") == "262144"
    assert arg(argv, "--gpu-memory-utilization") == "0.93"
    assert arg(argv, "--kv-cache-dtype") == "fp8"
    assert "--enable-auto-tool-choice" in argv and arg(argv, "--reasoning-parser") == "qwen3"
    assert env["B70_TARGET_LMHEAD_FP8"] == "1"
    spec = json.loads(arg(argv, "--speculative-config"))
    assert spec["method"] == "mtp" and spec["num_speculative_tokens"] == 4
    assert "--enforce-eager" not in argv


def test_capture_sizes_follow_mtp_k(run):
    for k in (1, 4, 9):
        r, out = run({"B70_MTP_K": str(k)})
        assert r.returncode == 0, r.stderr
        graphs = json.loads(arg(out["argv"], "--compilation-config"))["cudagraph_capture_sizes"]
        assert graphs == sorted(graphs) and len(graphs) == len(set(graphs))
        # every batch of 1..8 requests, each verifying k+1 tokens, must hit a captured size
        assert all((k + 1) * n in graphs for n in range(1, 9))
        assert {1, 2, 4, 8} <= set(graphs)
        assert json.loads(arg(out["argv"], "--speculative-config"))["num_speculative_tokens"] == k


@pytest.mark.parametrize("bad", ["0", "10", "x", "", "4 "])
def test_mtp_k_is_validated(run, bad):
    r, out = run({"B70_MTP_K": bad})
    if bad == "":  # an empty value falls back to the default of 4
        assert r.returncode == 0
        return
    assert r.returncode == 2 and "B70_MTP_K must be 1-9" in r.stderr
    assert out is None  # nothing was launched


def test_fast_off_is_eager_without_mtp(run):
    r, out = run({"FAST": "0"})
    assert r.returncode == 0, r.stderr
    assert "--enforce-eager" in out["argv"]
    assert "--speculative-config" not in out["argv"] and "--compilation-config" not in out["argv"]
    assert out["env"]["VLLM_XPU_ENABLE_XPU_GRAPH"] == "0"


def test_env_overrides_and_extra_args(run):
    r, out = run({"PORT": "9001", "SERVED_NAME": "x", "GPU_UTIL": "0.8", "MAX_MODEL_LEN": "4096"},
                 args=("--language-model-only",))
    argv = out["argv"]
    assert (arg(argv, "--port"), arg(argv, "--served-model-name")) == ("9001", "x")
    assert arg(argv, "--gpu-memory-utilization") == "0.8" and arg(argv, "--max-model-len") == "4096"
    assert argv[-1] == "--language-model-only"  # extra args come last, so they can override earlier flags


def test_prefix_unit_is_opt_in(run):
    _, out = run()
    assert "--prefix-match-unit" not in out["argv"]
    _, out = run({"B70_PREFIX_UNIT": "128"})
    assert arg(out["argv"], "--prefix-match-unit") == "128"


def test_linear_fp8_regex(run):
    _, out = run()
    pat = re.compile(out["env"]["B70_LINEAR_FP8_RE"])
    for name in ("model.layers.3.self_attn.qkv_proj", "model.layers.3.self_attn.o_proj",
                 "model.layers.0.linear_attn.in_proj_qkvz", "model.layers.0.linear_attn.out_proj",
                 "model.layers.1.mlp.shared_expert.gate_up_proj", "model.layers.1.mlp.shared_expert.down_proj"):
        assert pat.search(name), name
    for name in ("lm_head", "model.layers.1.mlp.experts.w13", "model.layers.2.mlp.gate",
                 "model.layers.3.self_attn.qkv_proj.weight", "mtp.layers.0.self_attn.o_proj_extra"):
        assert not pat.search(name), name
    _, out = run({"B70_LINEAR_FP8": "0"})
    assert "B70_LINEAR_FP8_RE" not in out["env"]
    _, out = run({"B70_LINEAR_FP8_RE": "custom"})
    assert out["env"]["B70_LINEAR_FP8_RE"] == "custom"
    _, out = run({"B70_TARGET_LMHEAD_FP8": "0"})
    assert out["env"]["B70_TARGET_LMHEAD_FP8"] == "0"


def test_jev_checkpoint_switches_setup(run):
    r, out = run(jev=True)
    assert r.returncode == 0, r.stderr
    argv, env = out["argv"], out["env"]
    assert out["cmd"] == "serve_decide"
    assert env["B70_TARGET_LMHEAD_FP8"] == "0"  # LoRA buffers follow the lm_head dtype
    assert env["JEV_DECIDE_CALIBRATION"].endswith("calibration.json")
    assert arg(argv, "--max-num-seqs") == "8"  # JEV requires 8
    assert arg(argv, "--max-model-len") == "131072" and arg(argv, "--gpu-memory-utilization") == "0.90"
    assert arg(argv, "--served-model-name") == "autotrust/JEV-27B-VL"
    assert "--enable-lora" in argv and "--enable-auto-tool-choice" not in argv
    assert "B70_LINEAR_FP8_RE" not in env
