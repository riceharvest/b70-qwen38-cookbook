#!/usr/bin/env python3
"""B70: drop the two host syncs per MTP decode step in the GDN metadata builder (backport from vLLM main).

In v0.31.0 the pure spec-decode branch of GDNAttentionMetadataBuilder.build() gathers two device tensors with the host
bool mask `spec_sequence_masks_cpu` (block_table_tensor[mask, :k] and num_accepted_tokens[mask]). PyTorch turns a host
mask into indices and copies them to the device with a blocking copy, which waits for all queued GPU work: on the B70
that drains the queue (measured 284 of 288 ms of queued GEMMs) twice on every MTP decode step, so async scheduling
loses its CPU/GPU overlap. Padded rows trail the spec rows (the existing `query_start_loc[: num_spec_decodes + 1]`
already relies on that), so both gathers are plain slices, as on vLLM main. Same values; `.contiguous()` keeps the
state-index layout the mask gather produced (a device copy, no sync).

Runs after sergiiob/patch_mtp_boundary.py (it anchors on that patch's text). Mixed prefill+spec steps keep their
mask gathers (fixed on vLLM main: riceharvest/vllm hunt-gdn-meta-no-mask-sync).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MARKER = "B70_GDN_SPEC_NO_MASK_SYNC"

OLD = '''                    # Filter by spec_sequence_masks to exclude padded sequences
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
'''

NEW = '''                    # B70_GDN_SPEC_NO_MASK_SYNC: padded sequences trail the spec
                    # decodes, so slice them off rather than gather with the host
                    # mask (a blocking H2D index copy = a full device sync).
                    spec_state_indices_tensor = block_table_tensor[
                        :num_spec_decodes, : self.num_spec + 1
                    ].contiguous()
'''

OLD_FINALIZE = '''            if spec_sequence_masks_cpu is not None:
                assert num_accepted_tokens is not None
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
'''

NEW_FINALIZE = '''            if spec_sequence_masks_cpu is not None:
                assert num_accepted_tokens is not None
                if num_prefills == 0 and num_decodes == 0:
                    num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
                else:
                    num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
'''


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    if text.count(OLD) != 1:
        raise RuntimeError("GDN pure-spec state-index gather anchor changed (needs patch_mtp_boundary first); refusing")
    if text.count(OLD_FINALIZE) != 1:
        raise RuntimeError("GDN num_accepted_tokens gather anchor changed (needs patch_mtp_boundary first); refusing")
    text = text.replace(OLD, NEW, 1)
    return text.replace(OLD_FINALIZE, NEW_FINALIZE, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = Path(next(iter(spec.submodule_search_locations))) / "v1/attention/backends/gdn_attn.py"
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    original = path.read_text()
    patched = patch_text(original)
    if patched == original:
        print(f"already patched {path}")
        return
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
