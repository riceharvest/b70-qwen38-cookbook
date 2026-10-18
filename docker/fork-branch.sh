#!/usr/bin/env bash
# Rebuild the vLLM fork branch (riceharvest/vllm b70/v<ver>) = the vLLM release tag + one commit per cookbook patch, in the
# Dockerfile's order. The patches stay canonical; this branch is the same code as git history, for working on it as source.
# usage: docker/fork-branch.sh VLLM_CHECKOUT [--push]   (VLLM_CHECKOUT: a vllm clone with remote `fork` = riceharvest/vllm)
# env: IMG (the built image to check against, default the Dockerfile's tag)
set -euo pipefail
cd "$(dirname "$0")/.."; CB=$PWD
SRC=$(realpath "$1"); PUSH=${2:-}
BASE_IMG=$(sed -n 's/^FROM //p' docker/Dockerfile)
VER=$(echo "$BASE_IMG" | sed -n 's/.*:\(v[0-9.]*\)@.*/\1/p'); BRANCH=b70/${VER%.0}
IMG=${IMG:-ghcr.io/riceharvest/b70-vllm:$(sed -n 's/^TAG=${TAG:-\([^}]*\)}.*/\1/p' docker/build.sh)}
SP=/opt/venv/lib/python3.12/site-packages
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
# patch order = the Dockerfile RUN loops
python3 - docker/Dockerfile > "$WORK/order" <<'EOF'
import re, sys
for loop in re.finditer(r'for p in ([^;]+); do python /b70/patches/(\w+)/\$p\.py', open(sys.argv[1]).read()):
    for p in loop.group(1).replace('\\', ' ').split():
        print(loop.group(2), p)
EOF
# apply the patches one by one on the stock base image; after each, save the files it changed (vllm + vllm_xpu_kernels)
mkdir -p "$WORK/out"
docker run --rm --entrypoint bash -v "$PWD/patches:/b70/patches:ro" -v "$PWD/runtime:/opt/b70:ro" -v "$WORK:/w" "$BASE_IMG" -lc "
set -e; cd $SP
sums() { find vllm vllm_xpu_kernels -type f ! -path '*/__pycache__/*' ! -name '*.before_*' ! -name '*.bak' ! -name '*.orig' -print0 | sort -z | xargs -0 md5sum; }
sums > /w/prev; n=0
while read dir p; do
  n=\$((n+1)); step=/w/out/\$(printf %02d \$n)-\$dir-\$p; mkdir -p \$step
  python /b70/patches/\$dir/\$p.py > \$step.log 2>&1 || { cat \$step.log; exit 1; }
  sums > /w/cur
  diff /w/prev /w/cur | sed -n 's/^> [0-9a-f]*  //p' | while read f; do mkdir -p \$step/\$(dirname \$f); cp \$f \$step/\$f; done
  diff /w/prev /w/cur | sed -n 's/^< [0-9a-f]*  //p' | while read f; do grep -q \"  \$f\$\" /w/cur || echo \$f >> \$step.removed; done
  mv /w/cur /w/prev
done < /w/order
chown -R $(id -u):$(id -g) /w"
# commit them on top of the release tag
cd "$SRC"
git fetch -q origin tag "$VER" --no-tags; git fetch -q fork main
TREE=$WORK/tree; git worktree prune; trap 'git -C "$SRC" worktree remove --force "$TREE" 2>/dev/null; rm -rf "$WORK"' EXIT; git worktree add -q -B "$BRANCH" "$TREE" "$VER"
for step in "$WORK"/out/*/; do
  echo "$(basename "$step"): $(cd "$step" && find . -type f | wc -l) files"
  name=$(basename "$step"); dir=$(echo "$name" | cut -d- -f2); p=${name#*-*-}
  [ -d "$step/vllm" ] && cp -r "$step/vllm" "$TREE/"
  [ -f "${step%/}.removed" ] && grep '^vllm/' "${step%/}.removed" | (cd "$TREE" && xargs -r git rm -q)
  xk=$( (cd "$step" && find vllm_xpu_kernels -type f 2>/dev/null || true) | sed 's/^/  /')
  (cd "$TREE" && git add -A vllm && git diff --cached --quiet) && [ -z "$xk" ] && { echo "skip $p (no change)"; continue; }
  head=$(python3 -c "import ast,sys; d=ast.get_docstring(ast.parse(open(sys.argv[1]).read())) or ''; print(d.splitlines()[0].removeprefix('B70: ') if d else '')" "$CB/patches/$dir/$p.py")
  msg="[B70] $p: $head"$'\n\n'"Generated from patches/$dir/$p.py of riceharvest/b70-qwen38-cookbook@$(git -C "$CB" rev-parse --short HEAD)."
  [ -n "$xk" ] && msg+=$'\n'"The patch also edits vllm-xpu-kernels files, not carried here:"$'\n'"$xk"
  [ "$dir" = sergiiob ] && msg+=$'\n\n'"Co-authored-by: SergiioB <119968745+SergiioB@users.noreply.github.com>"
  (cd "$TREE" && git commit -q --allow-empty -m "$msg")
done
# equivalence: the branch's vllm/ (python sources) must match the shipped image byte for byte
c=$(docker create "$IMG"); docker cp -q "$c:$SP/vllm" "$WORK/img"; docker rm "$c" >/dev/null
if diff -rq -x __pycache__ -x '*.so' -x _version.py -x vllm-rs "$TREE/vllm" "$WORK/img" | grep -v '^Only in /.*/img'; then
  echo "MISMATCH between $BRANCH and $IMG" >&2; exit 1
fi
echo "$BRANCH = $VER + $(git -C "$TREE" rev-list --count "$VER"..HEAD) commits; vllm/ matches $IMG"
git worktree remove --force "$TREE"
[ "$PUSH" = --push ] && git push -f fork "$BRANCH"
