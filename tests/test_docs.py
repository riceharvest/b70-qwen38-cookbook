"""The docs stay true to the tree: relative links resolve, and repo paths named in backticks exist."""
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
DOCS = sorted(p for p in REPO.rglob("*.md") if not {"build", ".git", "node_modules"} & set(p.relative_to(REPO).parts))
PATH_RE = re.compile(r"`((?:kernels|docker|runtime|quant|bench|docs|tests|licenses)/[\w./-]+)`")
LINK_RE = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_relative_links_resolve(doc):
    bad = []
    for m in LINK_RE.finditer(doc.read_text()):
        url = m.group(1)
        if re.match(r"[a-z][a-z0-9+.-]*:", url) or url.startswith("../../"):  # absolute, or a GitHub web path (../../releases)
            continue
        if not (doc.parent / url).exists():
            bad.append(url)
    assert not bad, f"{doc.name}: broken relative links {bad}"


def _gitignored(path):
    """Build outputs the docs name (a prebuilt .so, the corpus) are gitignored, so a fresh checkout doesn't have them."""
    try:
        return subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=REPO, check=False).returncode == 0
    except OSError:
        return False


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_named_repo_paths_exist(doc):
    # paths in docs/ and README refer to the repo root; a doc deeper in the tree may also use paths relative to itself
    bad = [p for p in dict.fromkeys(PATH_RE.findall(doc.read_text()))
           if not (REPO / p.rstrip("/")).exists() and not (doc.parent / p.rstrip("/")).exists()
           and not _gitignored(REPO / p.rstrip("/"))]
    assert not bad, f"{doc.name}: paths that don't exist {bad}"


def _slug(heading):
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_in_page_anchors_exist(doc):
    text = doc.read_text()
    text_no_code = re.sub(r"```.*?```", "", text, flags=re.S)  # '# comment' lines in code blocks are not headings
    slugs = {_slug(m.group(2)) for m in re.finditer(r"^(#+) (.*)$", text_no_code, re.M)}
    bad = [a for a in re.findall(r"\]\(#([^)]+)\)", text) if a not in slugs]
    assert not bad, f"{doc.name}: anchors without a heading {bad}"
