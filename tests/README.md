# Patch tests (CPU only)

`test_patches.py` applies every script in `patches/` to the vLLM sources of the pinned base image, in the Dockerfile's
order, in a temp copy. It checks that:

- the Dockerfile applies every patch script exactly once
- each script exits 0 and changes at least one file
- the patched files compile and a second run changes nothing
- each script fails (exit != 0) on a tree without vLLM, so a vLLM bump that moves a target breaks the image build
  instead of shipping it unpatched (SergiioB's accept-sync patch warns by design and is skipped there)

```bash
tests/fetch_vllm_src.sh          # once per base image: copies the .py files of vllm + vllm_xpu_kernels out of it (~40 MB, no GPU)
python -m pytest tests -q        # ~10 s; without the sources only the static checks run
```

`B70_VLLM_SRC=/path/to/site-packages` points the tests at another tree, e.g. a newer vLLM before bumping the image.
