"""CPU-only tests for the patch scripts in patches/: they apply cleanly, in the Dockerfile's order, to the vLLM sources of
the pinned base image, a second run changes nothing, and every patched file still compiles.

The sources come from tests/fetch_vllm_src.sh (default ~/.cache/b70-cookbook-tests/vllm-xpu-v0.31.0/site-packages, or
$B70_VLLM_SRC). Without them only the static checks run. Run: python -m pytest tests -q
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
SRC = pathlib.Path(os.environ.get("B70_VLLM_SRC",
                                  pathlib.Path.home() / ".cache/b70-cookbook-tests/vllm-xpu-v0.31.0/site-packages"))
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
