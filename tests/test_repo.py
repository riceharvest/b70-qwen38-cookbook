"""Every Python file in the repo compiles (no imports: the bench/quant scripts need torch / a GPU to run)."""
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
import subprocess


def _python_files():
    try:
        out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
        if out:
            return sorted(REPO / f for f in out)
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(p for p in REPO.rglob("*.py") if not {".git", ".venv", "venv", "build"} & set(p.relative_to(REPO).parts))


FILES = _python_files()


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_compiles(path):
    compile(path.read_text(), str(path), "exec")
