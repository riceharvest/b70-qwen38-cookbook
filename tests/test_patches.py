"""CPU-only tests for the patch scripts in patches/: they apply cleanly, in the Dockerfile's order, to the vLLM sources of
the pinned base image, a second run changes nothing, and every patched file still compiles.

The sources come from tests/fetch_vllm_src.sh (default ~/.cache/b70-cookbook-tests/<hash of the FROM line>/site-packages,
or $B70_VLLM_SRC). Without them only the static checks run. Run: python -m pytest tests -q
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
PATCHES = REPO / "patches"
SITE = "/opt/venv/lib/python3.12/site-packages"


def _base_image():
    m = re.search(r"^FROM (\S+)", (REPO / "docker" / "Dockerfile").read_text(), re.M)
    return m.group(1)


# same key as tests/fetch_vllm_src.sh: a new FROM means new sources, never a stale cache
_KEY = hashlib.sha256(_base_image().encode()).hexdigest()[:12]
SRC = pathlib.Path(os.environ.get("B70_VLLM_SRC",
                                  pathlib.Path.home() / ".cache/b70-cookbook-tests" / _KEY / "site-packages"))
HAVE_SRC = (SRC / "vllm" / "__init__.py").is_file() and (SRC / "vllm_xpu_kernels").is_dir()
needs_src = pytest.mark.skipif(not HAVE_SRC, reason=f"no vLLM sources at {SRC} (run tests/fetch_vllm_src.sh)")


def dockerfile_order():
    """[(dir, name), ...] in the order the Dockerfile's RUN applies them."""
    text = (REPO / "docker" / "Dockerfile").read_text().replace("\\\n", " ")
    order = []
    for loop in re.finditer(r"for p in ([\w ]+); do python /b70/patches/(\w+)/\$p\.py", text):
        order += [(loop.group(2), name) for name in loop.group(1).split()]
    return order


ORDER = dockerfile_order()


def test_dockerfile_lists_every_patch_once():
    on_disk = sorted((p.parent.name, p.stem) for p in PATCHES.glob("*/patch_*.py"))
    assert len(ORDER) == len(set(ORDER)), "a patch is applied twice"
    assert sorted(ORDER) == on_disk


@pytest.mark.parametrize("script", sorted(PATCHES.glob("*/patch_*.py")), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_script_compiles(script):
    compile(script.read_text(), str(script), "exec")


def _tree_hashes(root):
    return {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.py")}


def _run(script, sandbox):
    """Run one patch script against the sandbox: its hard-coded site-packages path is rewritten to the sandbox, and
    `import vllm` (in-process or in a child python) finds the sandbox's stubbed vllm package."""
    text = script.read_text().replace(SITE, str(sandbox))
    tmp = sandbox.parent / "scripts" / script.parent.name / script.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(sandbox), "PYTHONDONTWRITEBYTECODE": "1",
           "B70_VLLM_ROOTS": str(sandbox / "vllm")}
    return subprocess.run([sys.executable, "-s", str(tmp)], cwd=sandbox.parent, env=env, capture_output=True, text=True,
                          timeout=300)


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    """A copy of the image's sources with every patch applied once, plus the first-run outputs and pre-patch hashes."""
    base = tmp_path_factory.mktemp("b70patch")
    sandbox = base / "site-packages"
    shutil.copytree(SRC, sandbox)
    # the real vllm/__init__.py imports torch; the scripts only need vllm.__file__ / find_spec to locate the tree
    (sandbox / "vllm" / "__init__.py").write_text("# test stub\n")
    before = _tree_hashes(sandbox)
    runs, touched = {}, {}
    for d, n in ORDER:
        h0 = _tree_hashes(sandbox)
        runs[f"{d}/{n}"] = _run(PATCHES / d / f"{n}.py", sandbox)
        h1 = _tree_hashes(sandbox)
        touched[f"{d}/{n}"] = [p for p in h1 if h0.get(p) != h1[p]]
    return sandbox, before, runs, touched


@needs_src
@pytest.mark.parametrize("key", [f"{d}/{n}" for d, n in ORDER])
def test_patch_applies(patched, key):
    _, _, runs, touched = patched
    r = runs[key]
    assert r.returncode == 0, f"{key} rc={r.returncode}\nstdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-2000:]}"
    assert touched[key], f"{key} exited 0 but changed no file:\n{r.stdout[-1000:]}"


@needs_src
def test_patched_files_compile(patched):
    sandbox, before, _, _ = patched
    after = _tree_hashes(sandbox)
    bad = []
    for rel in (p for p in after if before.get(p) != after[p]):
        try:
            compile((sandbox / rel).read_text(), str(rel), "exec")
        except SyntaxError as e:
            bad.append(f"{rel}: {e}")
    assert not bad, "\n".join(bad)


@needs_src
def test_second_run_is_a_noop(patched):
    sandbox, _, _, _ = patched
    first = _tree_hashes(sandbox)
    failed = []
    for d, n in ORDER:
        r = _run(PATCHES / d / f"{n}.py", sandbox)
        if r.returncode != 0:
            failed.append(f"{d}/{n} rc={r.returncode}: {(r.stdout + r.stderr)[-500:]}")
    assert not failed, "\n".join(failed)
    second = _tree_hashes(sandbox)
    assert first == second, f"changed on the second run: {sorted(str(p) for p in second if first.get(p) != second[p])}"


# Scripts that exit 0 on purpose when their target is missing (SergiioB's accept-sync: the patched path is not used in
# this image's config, so it warns instead of failing the build).
WARN_ONLY = {"sergiiob/patch_fix_accepted_sync"}


@pytest.mark.parametrize("key", [f"{d}/{n}" for d, n in ORDER])
def test_fails_loudly_without_vllm(tmp_path, key):
    """A vLLM bump that moves or renames a target must break the image build, not ship it unpatched."""
    if key in WARN_ONLY:
        pytest.skip("warns by design")
    sandbox = tmp_path / "site-packages"
    (sandbox / "vllm").mkdir(parents=True)
    (sandbox / "vllm_xpu_kernels").mkdir()
    (sandbox / "vllm" / "__init__.py").write_text("# test stub\n")
    d, n = key.split("/")
    r = _run(PATCHES / d / f"{n}.py", sandbox)
    assert r.returncode != 0, f"{key} exited 0 on an empty tree:\n{(r.stdout + r.stderr)[-1000:]}"


@needs_src
def test_gdn_spec_sync_slices_pure_spec_branch(patched):
    """patch_gdn_spec_sync: the pure MTP-decode branch slices instead of gathering with the host mask (each host-mask
    gather of a device tensor is a blocking H2D copy = a full GPU queue drain); the mixed branch keeps its gathers."""
    sandbox, _, _, _ = patched
    rel = "vllm/v1/attention/backends/gdn_attn.py"
    orig, new = (SRC / rel).read_text(), (sandbox / rel).read_text()
    assert "B70_GDN_SPEC_NO_MASK_SYNC" in new
    assert "B70_MTP_PARTIAL_FINAL_GROUP" in new  # applied on top of sergiiob/patch_mtp_boundary
    pure = new[new.index("B70_GDN_SPEC_NO_MASK_SYNC"):]
    pure = pure[:pure.index("            else:\n")]
    assert "block_table_tensor[\n                        :num_spec_decodes, : self.num_spec + 1\n" in pure
    assert "spec_sequence_masks_cpu" not in pure
    assert "num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]" in new
    # the pure-spec state-index gather is gone (its text survives nowhere); the mixed step is patch_gdn_mixed_no_sync's
    # (whose switch-off branch keeps the original mask gathers, so a whole-file gather count says nothing here)
    pure_gather = re.compile(r"spec_sequence_masks_cpu, : self\.num_spec \+ 1\s*\]\s*non_spec_state_indices_tensor = None")
    assert pure_gather.search(orig) and not pure_gather.search(new)
    compile(new, rel, "exec")


@needs_src
def test_gdn_zero_draft_keeps_decode_rows_speculative(patched):
    """patch_gdn_zero_draft: V2 marks every decode row (0..k drafts, width <= k+1) speculative, the builder no longer
    falls back to the non-spec path for an all-zero-draft batch or a partial pure-spec group (ragged spec instead)."""
    sandbox, _, _, _ = patched
    hybrid = (sandbox / "vllm/v1/worker/gpu/model_states/mamba_hybrid.py").read_text()
    gdn = (sandbox / "vllm/v1/attention/backends/gdn_attn.py").read_text()
    assert "B70_GDN_ZERO_DRAFT" in hybrid and "B70_GDN_ZERO_DRAFT" in gdn
    assert "num_draft_tokens_per_req = np.zeros(input_batch.num_reqs, np.int32)" in hybrid
    assert "<= self.vllm_config.num_speculative_tokens + 1" in hybrid  # #55504 width bound
    assert "is_decode, num_draft_tokens_per_req, -1" in hybrid
    assert "def _b70_zero_draft_on()" in hybrid
    # all-zero-draft fallback only with the switch off; partial groups stay speculative with the real token count
    assert 'os.environ.get("B70_GDN_ZERO_DRAFT", "1") == "0"\n                and num_decode_draft_tokens_cpu' in gdn
    assert "query_start_loc_cpu[num_spec_decodes].item()" in gdn
    assert "actual_spec_token_size\n                        if _b70_ragged" in gdn
    assert "\nimport os\n" in gdn
    compile(hybrid, "mamba_hybrid.py", "exec")
    compile(gdn, "gdn_attn.py", "exec")


@needs_src
def test_gdn_mixed_no_sync_uses_index_select(patched):
    """patch_gdn_mixed_no_sync: the mixed step gathers with device row indices from one async copy; the host-mask
    gathers survive only in the B70_GDN_MIXED_NO_SYNC=0 branch."""
    sandbox, _, _, _ = patched
    gdn = (sandbox / "vllm/v1/attention/backends/gdn_attn.py").read_text()
    assert "B70_GDN_MIXED_NO_SYNC" in gdn
    assert len(re.findall(r"index_select\(\s*0, _b70_spec_rows\s*\)", gdn)) == 2  # state indices + num_accepted
    assert len(re.findall(r"index_select\(\s*0, _b70_non_spec_rows\s*\)", gdn)) == 2  # state indices + has_initial
    assert "_b70_packed = async_tensor_h2d(" in gdn
    on = gdn[gdn.index('if os.environ.get("B70_GDN_MIXED_NO_SYNC", "1") != "0":'):]
    on = on[:on.index("                else:\n")]
    assert "masks_cpu]" not in on.replace("query_lens_cpu[spec_sequence_masks_cpu]", "").replace(
        "query_lens_cpu[non_spec_sequence_masks_cpu]", "")
    compile(gdn, "gdn_attn.py", "exec")


@needs_src
def test_xpu_graph_kv_splits_passes_num_splits(patched):
    sandbox, _, _, _ = patched
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
def test_spec_greedy_warmup_runs_second_pass(patched):
    sandbox, _, _, _ = patched
    w = (sandbox / "vllm/v1/worker/gpu/warmup.py").read_text()
    assert "B70_SPEC_GREEDY_WARMUP" in w
    assert "worker_sample_tokens, greedy=True" in w
    assert "SamplingParams(temperature=0)" in w
    compile(w, "warmup.py", "exec")


@needs_src
def test_eager_token_pad_buckets_none_mode(patched):
    sandbox, _, _, _ = patched
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
