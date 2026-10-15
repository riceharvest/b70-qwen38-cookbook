"""B70: route embed_tokens through the UVA offloader (weights live in pinned host RAM,
GPU gathers rows zero-copy over PCIe). Needs --cpu-offload-gb >= 2.4 and
--cpu-offload-params embed_tokens [visual]. Gate: B70_EMBED_UVA=1 (default on once patched)."""
import pathlib, sys
MARK = "# B70_EMBED_UVA"
roots = ["/opt/venv/lib/python3.12/site-packages/vllm", "/workspace/vllm/vllm"]
n = 0
for r in roots:
    for fn in ["model_executor/models/qwen3_5.py", "model_executor/models/qwen3_5_mtp.py"]:
        p = pathlib.Path(r) / fn
        if not p.exists(): continue
        s = p.read_text()
        if MARK in s: n += 1; continue
        i = s.find("self.embed_tokens = VocabParallelEmbedding(")
        if i < 0: print("anchor missing", p); sys.exit(1)
        # end of the constructor call: first line that is just ')' at the same indent
        line_start = s.rfind("\n", 0, i) + 1; indent = s[line_start:i]
        j = s.find("\n" + indent + ")", i); j = s.find("\n", j + 1)
        inject = (f"\n{indent}{MARK}\n{indent}import os as _b70os\n"
                  f"{indent}if _b70os.environ.get('B70_EMBED_UVA', '1') == '1':\n"
                  f"{indent}    from vllm.model_executor.offloader.base import get_offloader as _b70go\n"
                  f"{indent}    _b70go().wrap_modules(iter([self.embed_tokens]), prefix=f\"{{prefix}}.embed_tokens\")")
        s = s[:j] + inject + s[j:]
        p.write_text(s); n += 1; print("patched", p)
print("embed-uva files:", n)
if n == 0: sys.exit("embed-uva: qwen3_5.py / qwen3_5_mtp.py not found in any vLLM tree")
