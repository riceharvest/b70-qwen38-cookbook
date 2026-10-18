#!/usr/bin/env python3
"""B70: no host-mask gathers in the GDN metadata builder's mixed step (spec decodes + prefills in one step).

vLLM v0.32 removed the two blocking gathers of the pure MTP-decode step (our old patch_gdn_spec_sync). Steps that
also carry a prefill (a new or long prompt while other requests decode: the normal state with agents and chunked
prefill) still gather device tensors with host bool masks: block_table x2, query_lens x2, num_accepted_tokens and has_initial_state.
PyTorch turns each host mask into indices and copies them to the device with a blocking copy, which waits for all
queued GPU work (measured on the B70: 305-309 ms per call with ~290 ms of GEMMs queued, vs 0.5 ms for this path).
Every mixed step therefore loses the async-scheduling CPU/GPU overlap.

Build the spec / non-spec row indices and both cumulative query lengths on the host (the query lens are already
there), send them in ONE async pinned H2D copy and gather with index_select. Same values, order, dtype and contiguity.
Backport of riceharvest/vllm hunt-gdn-meta-no-mask-sync (1a9e005) to the v0.32.0 text.
Switch off with B70_GDN_MIXED_NO_SYNC=0.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MARKER = "B70_GDN_MIXED_NO_SYNC"

INIT_OLD = '''        uniform_spec_sequence_length = None
        spec_sequence_masks_cpu: torch.Tensor | None = None
'''
INIT_NEW = '''        uniform_spec_sequence_length = None
        spec_sequence_masks_cpu: torch.Tensor | None = None
        # B70_GDN_MIXED_NO_SYNC: device row indices of the spec / non-spec rows
        # of a mixed step (None = use the host-mask gathers).
        _b70_spec_rows: torch.Tensor | None = None
        _b70_non_spec_rows: torch.Tensor | None = None
'''

MIXED_OLD = '''                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )
'''

MIXED_NEW = '''                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )
                if os.environ.get("B70_GDN_MIXED_NO_SYNC", "1") != "0":
                    # B70_GDN_MIXED_NO_SYNC: a host-mask gather of a device tensor
                    # is a blocking H2D index copy (= full device sync). Build the
                    # row indices and both cumsums on the host, one async copy.
                    _b70_spec_qsl_cpu = torch.zeros(
                        num_spec_decodes + 1, dtype=torch.int32
                    )
                    torch.cumsum(
                        query_lens_cpu[spec_sequence_masks_cpu],
                        dim=0,
                        out=_b70_spec_qsl_cpu[1:],
                    )
                    _b70_num_non_spec_rows = query_lens_cpu.size(0) - num_spec_decodes
                    _b70_packed = async_tensor_h2d(
                        torch.cat(
                            [
                                spec_sequence_masks_cpu.nonzero().flatten().int(),
                                non_spec_sequence_masks_cpu.nonzero().flatten().int(),
                                _b70_spec_qsl_cpu,
                                non_spec_query_start_loc_cpu,
                            ]
                        ),
                        device=query_start_loc.device,
                    )
                    (
                        _b70_spec_rows,
                        _b70_non_spec_rows,
                        spec_query_start_loc,
                        non_spec_query_start_loc,
                    ) = _b70_packed.split(
                        [
                            num_spec_decodes,
                            _b70_num_non_spec_rows,
                            num_spec_decodes + 1,
                            _b70_num_non_spec_rows + 1,
                        ]
                    )
                    spec_state_indices_tensor = block_table_tensor[
                        :, : self.num_spec + 1
                    ].index_select(0, _b70_spec_rows)
                    non_spec_state_indices_tensor = block_table_tensor[
                        :, 0
                    ].index_select(0, _b70_non_spec_rows)
                else:
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
                    non_spec_state_indices_tensor = block_table_tensor[
                        non_spec_sequence_masks_cpu, 0
                    ]
                    spec_query_start_loc = torch.zeros(
                        num_spec_decodes + 1,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    torch.cumsum(
                        query_lens[spec_sequence_masks_cpu],
                        dim=0,
                        out=spec_query_start_loc[1:],
                    )
                    non_spec_query_start_loc = torch.zeros(
                        query_lens.size(0) - num_spec_decodes + 1,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    torch.cumsum(
                        query_lens[non_spec_sequence_masks_cpu],
                        dim=0,
                        out=non_spec_query_start_loc[1:],
                    )
'''

ACC_OLD = '''            else:
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
'''
ACC_NEW = '''            elif _b70_spec_rows is not None:
                num_accepted_tokens = num_accepted_tokens.index_select(
                    0, _b70_spec_rows
                )
            else:
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
'''

HIS_OLD = '''                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
'''
HIS_NEW = '''                if _b70_non_spec_rows is not None:
                    has_initial_state = has_initial_state.index_select(
                        0, _b70_non_spec_rows
                    )
                else:
                    has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
'''


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    if "num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]" not in text:
        raise RuntimeError("gdn_attn.py lacks the pure-spec slicing of vLLM v0.32 (#56531 era); refusing")
    for old, what in ((INIT_OLD, "spec mask init"), (MIXED_OLD, "mixed-step gathers"),
                      (ACC_OLD, "num_accepted_tokens gather"), (HIS_OLD, "has_initial_state gather")):
        if text.count(old) != 1:
            raise RuntimeError(f"GDN {what} anchor changed or missing; refusing")
    text = text.replace(INIT_OLD, INIT_NEW, 1).replace(MIXED_OLD, MIXED_NEW, 1)
    text = text.replace(ACC_OLD, ACC_NEW, 1).replace(HIS_OLD, HIS_NEW, 1)
    if "\nimport os\n" not in text:
        if text.count("\nimport torch\n") != 1:
            raise RuntimeError("gdn_attn imports anchor changed; refusing")
        text = text.replace("\nimport torch\n", "\nimport os\n\nimport torch\n", 1)
    return text


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
