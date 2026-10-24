# Overlay tests (CPU only)

`test_overlay.py` exports the B70 fork changes (`docker/overlay.sh`, pins in `docker/fork.env`) and checks them against
the vLLM sources of the pinned base image:

- every file the overlay replaces is byte-identical in the image and at the fork base, so the overlay reverts nothing
  the image build changed (a vLLM bump with a stale fork base fails here), and no added file already exists
- the overlaid files compile, and the Dockerfile and `serve.sh` both apply the overlay
- content checks for single changes (GDN mixed-step no-sync, graph KV splits, greedy warmup, eager pad) on a copy of
  the image sources with the overlay applied, plus the runtime/ scripts

`test_serve_entrypoint.py` runs `docker/b70-serve.sh` against stub `python` / `vllm` binaries and checks what it would
launch: defaults, `B70_MTP_K` validation and the graph capture sizes that follow it, `FAST=0`, env overrides, the
fp8-linear regex, and the JEV switch. It needs no image and no sources.

`test_bench_scripts.py` runs `bench/harness.py` and `bench/arith_battery.py` against a fake vLLM server on localhost
(request shapes, tokens per MTP step and acceptance per position from the metric deltas, cache-defeating prefill prompts,
a seeded and stable arithmetic question set), and `test_tput_metric.py` feeds `bench/tput_metric.py` synthetic results and
server logs (it must flag runs where fewer than c requests ran at once). `test_docs.py` checks that links, paths and
anchors in the markdown files resolve.

```bash
tests/fetch_vllm_src.sh          # once per base image: copies the .py files of vllm + vllm_xpu_kernels out of it (~40 MB, no GPU)
python -m pytest tests -q        # ~10 s; without the sources only the static checks run
```

`B70_VLLM_SRC=/path/to/site-packages` points the tests at another tree, e.g. a newer vLLM before bumping the image.
