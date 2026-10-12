Copied unchanged from [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
at commit `1a76136ea1b3f96bba3b20e42041a6fcc95f56fd` (`patches/`), MIT License (see `LICENSE` here). Thanks, SergiioB.

They add MTP spec decode fixes for Qwen3.8 on vLLM XPU v0.31.0 (nightly MTP, boundary handling), int4 draft MTP layer and draft lm_head,
the GDN mixed-batch split, and three prefix-cache x MTP correctness fixes (accepted_sync, backward_copy, eagle_drop).
Apply them in the order listed in `serve.sh`.
