#!/usr/bin/env bash
# Export the B70 fork changes (docker/fork.env) as a site-packages overlay:
#   DEST/site-packages/{vllm,vllm_xpu_kernels}/...  the files each fork branch changes or adds against its base
#   DEST/base/...                                    the same files at the base commit (tests check the image matches them)
#   DEST/removed.txt                                 files the branches delete
# usage: docker/overlay.sh [DEST]   (default build/overlay; cached git objects in ~/.cache/b70-cookbook-forks)
set -euo pipefail
cd "$(dirname "$0")/.."
. docker/fork.env
DEST=${1:-build/overlay}; CACHE=${B70_FORK_CACHE:-$HOME/.cache/b70-cookbook-forks}
rm -rf "$DEST.partial"; mkdir -p "$DEST.partial/site-packages" "$DEST.partial/base"; : > "$DEST.partial/removed.txt"
export_diff() {  # name base_repo base fork_repo ref pkg_dir
  local g=$CACHE/$1
  [ -d "$g" ] || git init -q --bare "$g"
  git -C "$g" cat-file -e "$3^{commit}" 2>/dev/null || git -C "$g" fetch -q --depth 1 "$2" "$3"
  git -C "$g" cat-file -e "$5^{commit}" 2>/dev/null || git -C "$g" fetch -q --depth 1 "$4" "$5"
  git -C "$g" diff --no-renames --name-status "$3" "$5" -- "$6/" | while read -r st f; do
    case $st in
      D) echo "$f" >> "$DEST.partial/removed.txt" ;;
      A) mkdir -p "$DEST.partial/site-packages/$(dirname "$f")"; git -C "$g" show "$5:$f" > "$DEST.partial/site-packages/$f" ;;
      M) mkdir -p "$DEST.partial/site-packages/$(dirname "$f")" "$DEST.partial/base/$(dirname "$f")"
         git -C "$g" show "$5:$f" > "$DEST.partial/site-packages/$f"; git -C "$g" show "$3:$f" > "$DEST.partial/base/$f" ;;
      *) echo "unexpected change $st $f" >&2; exit 1 ;;
    esac
  done
}
export_diff vllm "$VLLM_BASE_REPO" "$VLLM_BASE" "$VLLM_REPO" "$VLLM_REF" vllm
export_diff vllm-xpu-kernels "$XK_BASE_REPO" "$XK_BASE" "$XK_REPO" "$XK_REF" vllm_xpu_kernels
printf '%s %s\n%s %s\n' vllm "$VLLM_REF" vllm_xpu_kernels "$XK_REF" > "$DEST.partial/REFS"
rm -rf "$DEST"; mv "$DEST.partial" "$DEST"
echo "$(find "$DEST/site-packages" -type f | wc -l) files, $(wc -l < "$DEST/removed.txt") removed -> $DEST"
