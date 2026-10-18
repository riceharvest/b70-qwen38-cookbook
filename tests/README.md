# Overlay tests (CPU only)

`test_overlay.py` exports the B70 fork changes (`docker/overlay.sh`, pins in `docker/fork.env`) and checks them against
the vLLM sources of the pinned base image:

- every file the overlay replaces is byte-identical in the image and at the fork base, so the overlay reverts nothing
  the image build changed (a vLLM bump with a stale fork base fails here), and no added file already exists
- the overlaid files compile, and the Dockerfile and `serve.sh` both apply the overlay
- content checks for single changes (GDN mixed-step no-sync, graph KV splits, greedy warmup, eager pad) on a copy of
  the image sources with the overlay applied, plus the runtime/ scripts

```bash
tests/fetch_vllm_src.sh          # once per base image: copies the .py files of vllm + vllm_xpu_kernels out of it (~40 MB, no GPU)
python -m pytest tests -q        # ~10 s; without the sources only the static checks run
```

`B70_VLLM_SRC=/path/to/site-packages` points the tests at another tree, e.g. a newer vLLM before bumping the image.
