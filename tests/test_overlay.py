"""CPU-only tests for the B70 overlay (docker/overlay.sh: the files the riceharvest/vllm and vllm-xpu-kernels fork
branches in docker/fork.env change): the base image's files are exactly the fork base's, so copying the fork's files
over them reverts nothing; the overlaid files compile; the Dockerfile and serve.sh both apply it.

The image sources come from tests/fetch_vllm_src.sh (default ~/.cache/b70-cookbook-tests/<hash of the FROM line>/
site-packages, or $B70_VLLM_SRC). Without them only the static checks run. Run: python -m pytest tests -q
"""
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _base_image():
    m = re.search(r"^FROM (\S+)", (REPO / "docker" / "Dockerfile").read_text(), re.M)
    return m.group(1)


# same key as tests/fetch_vllm_src.sh: a new FROM means new sources, never a stale cache
_KEY = hashlib.sha256(_base_image().encode()).hexdigest()[:12]
SRC = pathlib.Path(os.environ.get("B70_VLLM_SRC",
                                  pathlib.Path.home() / ".cache/b70-cookbook-tests" / _KEY / "site-packages"))
HAVE_SRC = (SRC / "vllm" / "__init__.py").is_file() and (SRC / "vllm_xpu_kernels").is_dir()
needs_src = pytest.mark.skipif(not HAVE_SRC, reason=f"no vLLM sources at {SRC} (run tests/fetch_vllm_src.sh)")


def _fork_env():
    return dict(re.findall(r"^(\w+)=(\S+)", (REPO / "docker" / "fork.env").read_text(), re.M))


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    """docker/overlay.sh output: site-packages/ (fork files), base/ (same files at the fork base), removed.txt."""
    dest = tmp_path_factory.mktemp("b70overlay") / "overlay"
    subprocess.run([str(REPO / "docker" / "overlay.sh"), str(dest)], check=True, capture_output=True, timeout=600)
    return dest


@pytest.fixture(scope="module")
def overlaid(overlay, tmp_path_factory):
    """A copy of the image's sources with the overlay applied, as the Dockerfile does."""
    if not HAVE_SRC:
        pytest.skip(f"no vLLM sources at {SRC}")
    sandbox = tmp_path_factory.mktemp("b70src") / "site-packages"
    shutil.copytree(SRC, sandbox)
    shutil.copytree(overlay / "site-packages", sandbox, dirs_exist_ok=True)
    for rel in (overlay / "removed.txt").read_text().split():
        (sandbox / rel).unlink()
    return sandbox


def test_overlay_is_the_pinned_refs(overlay):
    env = _fork_env()
    assert (overlay / "REFS").read_text().split() == ["vllm", env["VLLM_REF"], "vllm_xpu_kernels", env["XK_REF"]]
    files = sorted(p.relative_to(overlay / "site-packages").parts[0] for p in (overlay / "site-packages").rglob("*.py"))
    assert "vllm" in files and "vllm_xpu_kernels" in files


@needs_src
def test_image_matches_fork_base(overlay):
    """The overlay replaces whole files, so each must be byte-identical in the image and at the fork base; otherwise the
    overlay would silently revert what the image build changed. A vLLM bump with a mismatched base fails here."""
    base = overlay / "base"
    differ = [str(p.relative_to(base)) for p in base.rglob("*.py")
              if (SRC / p.relative_to(base)).read_bytes() != p.read_bytes()]
    assert not differ, f"image files differ from the fork base: {differ}"
    added = [p.relative_to(overlay / "site-packages") for p in (overlay / "site-packages").rglob("*.py")
             if not (base / p.relative_to(overlay / "site-packages")).exists()]
    clash = [str(p) for p in added if (SRC / p).exists()]
    assert not clash, f"files the fork adds already exist in the image: {clash}"


def test_overlaid_files_compile(overlay):
    bad = []
    for p in (overlay / "site-packages").rglob("*.py"):
        try:
            compile(p.read_text(), str(p), "exec")
        except SyntaxError as e:
            bad.append(f"{p}: {e}")
    assert not bad, "\n".join(bad)


def test_dockerfile_and_serve_apply_overlay():
    docker = (REPO / "docker" / "Dockerfile").read_text()
    serve = (REPO / "serve.sh").read_text()
    assert "COPY build/overlay/site-packages/ /opt/venv/lib/python3.12/site-packages/" in docker
    assert "build/overlay/removed.txt" in docker
    assert "docker/overlay.sh" in serve and "/b70/overlay/site-packages/." in serve and "/b70/overlay/removed.txt" in serve


@needs_src
def test_base_covers_dropped_gdn_patches():
    """vLLM v0.32 ships what patch_gdn_spec_sync, patch_gdn_zero_draft and sergiiob/patch_mtp_boundary did on v0.31
    (#56531, #55504 + the pure-spec slicing), so they are not on the fork branch. A base image without them must not ship."""
    hybrid = (SRC / "vllm/v1/worker/gpu/model_states/mamba_hybrid.py").read_text()
    gdn = (SRC / "vllm/v1/attention/backends/gdn_attn.py").read_text()
    # zero-draft decode rows stay speculative, width bound k+1 (#56531 + #55504)
    assert "num_draft_tokens_per_req = np.zeros(input_batch.num_reqs, np.int32)" in hybrid
    assert "<= self.vllm_config.num_speculative_tokens + 1" in hybrid
    assert "is_decode, num_draft_tokens_per_req, -1" in hybrid
    # an all-zero-draft batch keeps the speculative path
    assert "num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum()" not in gdn
    # pure-spec step: slices, no host-mask gathers; a partial final group stays speculative with its real token count
    assert "block_table_tensor[\n                    :num_spec_decodes, : self.num_spec + 1\n" in gdn
    assert "num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]" in gdn
    assert re.search(r"spec_token_size = min\(\s*num_spec_decodes \* \(self\.num_spec \+ 1\),\s*"
                     r"query_start_loc_cpu\[-1\]\.item\(\),", gdn)


@needs_src
def test_gdn_mixed_no_sync_uses_index_select(overlaid):
    """[B70] patch_gdn_mixed_no_sync: the mixed step gathers with device row indices from one async copy; the host-mask
    gathers survive only in the B70_GDN_MIXED_NO_SYNC=0 branch."""
    sandbox = overlaid
    gdn = (sandbox / "vllm/v1/attention/backends/gdn_attn.py").read_text()
    assert "B70_GDN_MIXED_NO_SYNC" in gdn
    assert len(re.findall(r"index_select\(\s*0, _b70_spec_rows\s*\)", gdn)) == 2  # state indices + num_accepted
    assert len(re.findall(r"index_select\(\s*0, _b70_non_spec_rows\s*\)", gdn)) == 2  # state indices + has_initial
    # the other GDN KV-cache groups re-gather with the same device rows (update_block_table)
    assert "b70_spec_rows=_b70_spec_rows," in gdn and "b70_non_spec_rows=_b70_non_spec_rows," in gdn
    regather = gdn[gdn.index("def update_block_table("):]
    regather = regather[:regather.index("\n    def ", 10)]
    assert re.search(r"index_select\(\s*0, m\.b70_spec_rows\s*\)", regather)
    assert re.search(r"index_select\(\s*0, m\.b70_non_spec_rows\s*\)", regather)
    assert "_b70_packed = async_tensor_h2d(" in gdn
    on = gdn[gdn.index('if os.environ.get("B70_GDN_MIXED_NO_SYNC", "1") != "0":'):]
    on = on[:on.index("                else:\n")]
    assert "masks_cpu]" not in on.replace("query_lens_cpu[spec_sequence_masks_cpu]", "").replace(
        "query_lens_cpu[non_spec_sequence_masks_cpu]", "")
    compile(gdn, "gdn_attn.py", "exec")


@needs_src
def test_xpu_graph_kv_splits_passes_num_splits(overlaid):
    sandbox = overlaid
    ops = (sandbox / "vllm/_xpu_ops.py").read_text()
    assert "def _b70_graph_decode_num_kv_splits(" in ops
    assert ops.count("**extra_kwargs,") >= 1
    assert "gpu_subslice_count" in ops
    compile(ops, "_xpu_ops.py", "exec")


def test_spec_packed_graph_split_rule():
    """runtime/spec_packed.py: one work-group per Xe core for the packed verify rows under graph capture."""
    import ast
    import types

    src = (REPO / "runtime" / "spec_packed.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_graph_splits")
    capturing = {"on": True}
    props = types.SimpleNamespace(gpu_subslice_count=32)
    fake_xpu = types.SimpleNamespace(is_current_stream_capturing=lambda: capturing["on"],
                                     get_device_properties=lambda i: props, current_device=lambda: 0)
    ns = {"os": os, "torch": types.SimpleNamespace(xpu=fake_xpu), "_XE_CORES": {}}
    exec(compile(ast.Module([fn], []), "spec_packed.py", "exec"), ns)
    q = types.SimpleNamespace(device=types.SimpleNamespace(index=0))
    rule = ns["_graph_splits"]
    # Qwen3.8-27B: Hkv 4, G 6, k=4 -> 30 packed rows per KV head, head_dim 256 -> 8-row tiles -> 4 WGs per KV head
    # measured optimum: ~256 WGs per call -> B=1/2/4/8 = 16/8/4/2 splits (031 sweep)
    assert [rule(q, b, 4, 30, 256) for b in (1, 2, 4, 8)] == [16, 8, 4, 2]
    assert rule(q, 16, 4, 30, 256) is None  # 256 WGs per split already: keep the kernel heuristic
    assert rule(q, 1, 2, 24, 128) == 32  # head_dim 128, group 24 -> 16-row tiles -> 2 WGs per KV head
    os.environ["B70_SPEC_Q32"] = "1"
    try:
        assert rule(q, 1, 4, 30, 256) == 32  # 32-row tile: one WG per KV head (cap 32)
    finally:
        del os.environ["B70_SPEC_Q32"]
    capturing["on"] = False
    assert rule(q, 1, 4, 30, 256) is None  # eager calls keep the heuristic
    capturing["on"] = True
    os.environ["B70_GRAPH_KV_SPLITS"] = "0"
    try:
        assert rule(q, 1, 4, 30, 256) is None
    finally:
        del os.environ["B70_GRAPH_KV_SPLITS"]


@needs_src
def test_spec_greedy_warmup_runs_second_pass(overlaid):
    sandbox = overlaid
    w = (sandbox / "vllm/v1/worker/gpu/warmup.py").read_text()
    assert "B70_SPEC_GREEDY_WARMUP" in w
    assert "worker_sample_tokens, greedy=True" in w
    assert "SamplingParams(temperature=0)" in w
    compile(w, "warmup.py", "exec")


@needs_src
def test_eager_token_pad_buckets_none_mode(overlaid):
    sandbox = overlaid
    r = (sandbox / "vllm/v1/worker/gpu/model_runner.py").read_text()
    assert "_B70_EAGER_PAD = int(" in r
    block = r[r.index("# B70_EAGER_PAD_TOKENS: bucket eager"):]
    block = block[:block.index("if batch_desc.num_tokens == 0:")]
    assert "batch_desc.cg_mode == CUDAGraphMode.NONE" in block and "not dummy_run" in block
    assert "self.max_num_tokens" in block  # never past the input buffers
    assert "batch_desc = replace(batch_desc, num_tokens=_b70_padded)" in block
    compile(r, "model_runner.py", "exec")


def test_inline_system_template(tmp_path):
    """runtime/inline_system_template.py: the Qwen3.8 template accepts a non-leading system message (vLLM's /v1/messages
    probe then keeps Claude Code's per-turn reminders in place instead of rewriting the system prompt every turn), and
    renders conversations without one byte-identically."""
    jinja2 = pytest.importorskip("jinja2")
    import jinja2.ext
    import jinja2.sandbox
    src = tmp_path / "model"
    src.mkdir()
    stock = (  # same structure as Qwen3.8: leading system rendered before the loop, the loop only guards
        "{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
        "{%- endif %}\n"
        "{%- for message in messages %}\n"
        "    {%- set content = message.content %}\n"
        '    {%- if message.role == "system" %}\n'
        "        {%- if not loop.first %}\n"
        "            {{- raise_exception('System message must be at the beginning.') }}\n"
        "        {%- endif %}\n"
        "    {%- else %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
    )
    (src / "chat_template.jinja").write_text(stock)
    out = tmp_path / "t.jinja"
    r = subprocess.run([sys.executable, str(REPO / "runtime" / "inline_system_template.py"), str(src), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == str(out)
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])

    def raise_exception(m):
        raise jinja2.TemplateError(m)

    env.globals["raise_exception"] = raise_exception
    probe = [{"role": "system", "content": "t"}, {"role": "user", "content": "t"},
             {"role": "system", "content": "t"}, {"role": "user", "content": "t"}]
    with pytest.raises(jinja2.TemplateError):
        env.from_string(stock).render(messages=probe)
    rendered = env.from_string(out.read_text()).render(messages=probe)
    assert rendered.count("<|im_start|>system") == 2
    plain = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert env.from_string(out.read_text()).render(messages=plain) == env.from_string(stock).render(messages=plain)
    # anchor missing -> no output, exit 0 (the launcher then keeps the model's template)
    (src / "chat_template.jinja").write_text("{{ messages }}")
    r = subprocess.run([sys.executable, str(REPO / "runtime" / "inline_system_template.py"), str(src), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == ""


def test_template_maps_openai_reasoning_effort(tmp_path):
    """runtime/inline_system_template.py: 'high' (Hermes' default) and 'minimal' no longer raise (HTTP 400); the
    supported values render unchanged."""
    jinja2 = pytest.importorskip("jinja2")
    import jinja2.ext
    import jinja2.sandbox
    src = tmp_path / "model"
    src.mkdir()
    stock = (
        "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n"
        "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
        "{{- raise_exception('Unexpected reasoning effort') }}{%- endif %}\n"
        "{{- resolved_reasoning_effort }}\n"
    )
    (src / "chat_template.jinja").write_text(stock)
    out = tmp_path / "t.jinja"
    r = subprocess.run([sys.executable, str(REPO / "runtime" / "inline_system_template.py"), str(src), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == str(out)
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])

    def raise_exception(m):
        raise jinja2.TemplateError(m)

    env.globals["raise_exception"] = raise_exception
    new = env.from_string(out.read_text())
    assert new.render(reasoning_effort="high").strip() == "xhigh"
    assert new.render(reasoning_effort="minimal").strip() == "low"
    for e in ("xhigh", "medium", "low"):
        assert new.render(reasoning_effort=e) == env.from_string(stock).render(reasoning_effort=e)
    with pytest.raises(jinja2.TemplateError):
        new.render(reasoning_effort="bogus")


@needs_src
def test_draft_int4_mse_is_opt_in(overlaid):
    """[B70] draft int4 MSE: both draft helpers carry the clip search, off unless B70_DRAFT_INT4_MSE=1 (same int4 layout)."""
    for name in ("b70_draft_mtp_int4.py", "b70_draft_lmhead_int4.py"):
        t = (overlaid / "vllm/model_executor/models" / name).read_text()
        assert "environ.get('B70_DRAFT_INT4_MSE', '0') == '1'" in t, name
        assert "for _b70_s in (1.0," in t and ".clamp(-8, 7).to(torch.int32)" in t
        compile(t, name, "exec")
