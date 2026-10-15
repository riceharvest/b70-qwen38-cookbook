"""Every Python file in the repo compiles (no imports: the bench/quant scripts need torch / a GPU to run)."""
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
FILES = sorted(p for p in REPO.rglob("*.py") if ".git" not in p.parts)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_compiles(path):
    compile(path.read_text(), str(path), "exec")
