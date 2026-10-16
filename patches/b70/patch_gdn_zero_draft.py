#!/usr/bin/env python3
"""B70: keep every MTP decode row on the GDN speculative path, including rows with 0 or fewer than k drafts
(backport of vLLM #56531 + #55504 for Model Runner V2, plus the matching change to the partial-group fallback).

The GDN speculative path keeps k+1 state slots per request and reads the initial state from slot
`num_accepted_tokens - 1`; the non-speculative path always reads slot 0. After a step that accepted drafts, the live
state sits in slot `accepted - 1`, so any row that switches to the non-speculative path for one step continues from a
state that is missing up to k tokens. Nothing raises; the output just goes wrong (upstream: 0/60 -> 60/60 tool calls).
Prefix caching (on by default) puts hybrids in mamba "align" mode, which only migrates the state at block boundaries,
so the stale read happens there too.

v0.31.0 switches paths in three places, all fixed here:
1. V2 `MambaHybridModelState.prepare_attn`: a decode row counts as speculative only with > 0 drafts, and a step with
   no scheduled drafts at all marks every row non-speculative. Now every decode row of width <= k+1 is speculative
   (0 drafts = a 1-token speculative row), prefill rows stay -1 (vLLM main, #56531 + #55504).
2. `GDNAttentionMetadataBuilder.build`: a batch whose speculative rows drafted nothing fell back to the non-spec path.
   Dropped, as on vLLM main.
3. `B70_MTP_PARTIAL_FINAL_GROUP` (sergiiob/patch_mtp_boundary): a pure-spec step with fewer than k drafts in any row
   ran the whole step through the non-spec prefill path, i.e. every row of that step read slot 0. The XPU GDN kernels
   in vllm-xpu-kernels 0.1.15.4 handle ragged speculative groups (#600: per-row spec_query_start_loc, capacity check
   `spec_token <= num_spec_decodes * (k + 1)`), so the step now stays speculative with the actual token count.

Runs after sergiiob/patch_mtp_boundary.py and b70/patch_gdn_spec_sync.py (anchors on their text).
Switch off with B70_GDN_ZERO_DRAFT=0 at runtime; B70_GDN_ZERO_DRAFT_LOG=1 logs how often 0-draft decode rows occur.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MARKER = "B70_GDN_ZERO_DRAFT"

# 1. V2 model state classification -------------------------------------------------------------------------------
HYBRID_OLD = '''            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                # Test request state, not num_scheduled_tokens == draft_count+1:
                # adaptive rewrites num_scheduled_tokens to an even split, so that
                # equality rarely holds and would demote every verify row to decode.
                is_decode = ~is_prefilling_np & (input_batch.num_scheduled_tokens > 0)
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask, num_draft_tokens_per_req, -1
                )
'''

HYBRID_NEW = '''            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if _b70_zero_draft_on():
                # B70_GDN_ZERO_DRAFT (vLLM #56531 + #55504): decode rows without
                # drafts stay speculative rows with 0 drafts. Only the speculative
                # path applies the previous step's accepted-token offset to the
                # recurrent state; slot 0 is stale after an accepted draft.
                if num_draft_tokens_per_req is None:
                    num_draft_tokens_per_req = np.zeros(input_batch.num_reqs, np.int32)
                is_decode = (
                    ~is_prefilling_np
                    & (input_batch.num_scheduled_tokens > 0)
                    & (
                        input_batch.num_scheduled_tokens
                        <= self.vllm_config.num_speculative_tokens + 1
                    )
                )
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    is_decode, num_draft_tokens_per_req, -1
                )
                if _B70_ZERO_DRAFT_LOG:
                    _b70_log_zero_draft_rows(
                        is_decode,
                        num_draft_tokens_per_req,
                        self.vllm_config.num_speculative_tokens,
                    )
            elif num_draft_tokens_per_req is not None:
                # Test request state, not num_scheduled_tokens == draft_count+1:
                # adaptive rewrites num_scheduled_tokens to an even split, so that
                # equality rarely holds and would demote every verify row to decode.
                is_decode = ~is_prefilling_np & (input_batch.num_scheduled_tokens > 0)
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask, num_draft_tokens_per_req, -1
                )
'''

HYBRID_HELPER_ANCHOR = "\nclass MambaHybridModelState("
HYBRID_HELPER = '''
def _b70_zero_draft_on() -> bool:
    # B70_GDN_ZERO_DRAFT: runtime switch (default on)
    import os

    return os.environ.get("B70_GDN_ZERO_DRAFT", "1") != "0"


# B70_GDN_ZERO_DRAFT_LOG=1: count decode rows with 0 drafts (host arrays only, no GPU reads); these are the rows the
# unpatched code sent down the non-speculative GDN path.
_B70_ZERO_DRAFT_LOG = __import__("os").environ.get("B70_GDN_ZERO_DRAFT_LOG") == "1"
_b70_zero_draft_counts = [0, 0, 0]  # [steps with a short row, zero-draft rows, rows with 1..k-1 drafts]


def _b70_log_zero_draft_rows(is_decode, num_draft_tokens_per_req, k) -> None:
    n = int((is_decode & (num_draft_tokens_per_req == 0)).sum())
    s = int((is_decode & (num_draft_tokens_per_req > 0) & (num_draft_tokens_per_req < k)).sum())
    if n or s:
        _b70_zero_draft_counts[0] += 1
        _b70_zero_draft_counts[1] += n
        _b70_zero_draft_counts[2] += s
        if _b70_zero_draft_counts[0] & (_b70_zero_draft_counts[0] - 1) == 0:  # 1, 2, 4, 8, ...
            print(
                f"[B70] short-draft decode rows: steps={_b70_zero_draft_counts[0]} "
                f"zero={_b70_zero_draft_counts[1]} short={_b70_zero_draft_counts[2]} (this step {n}/{s}, "
                f"decode rows {int(is_decode.sum())})",
                flush=True,
            )


'''

# 2 + 3. GDN metadata builder --------------------------------------------------------------------------------------
GDN_FALLBACK_OLD = '''            if (
                num_spec_decodes == 0
                or num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
                == 0
            ):
'''

GDN_FALLBACK_NEW = '''            # B70_GDN_ZERO_DRAFT: a batch whose rows all drafted nothing still
            # takes the speculative path (vLLM #56531).
            if num_spec_decodes == 0 or (
                os.environ.get("B70_GDN_ZERO_DRAFT", "1") == "0"
                and num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
                == 0
            ):
'''

GDN_PARTIAL_OLD = '''                if actual_spec_token_size < expected_spec_token_size:
                    # B70_MTP_PARTIAL_FINAL_GROUP:'''

GDN_PARTIAL_NEW = '''                # B70_GDN_ZERO_DRAFT: ragged speculative groups (0..k drafts per row)
                # stay on the speculative path; xk 0.1.15.4 traverses them by
                # spec_query_start_loc (#600). Count only the spec rows' tokens.
                _b70_ragged = os.environ.get("B70_GDN_ZERO_DRAFT", "1") != "0"
                if _b70_ragged:
                    actual_spec_token_size = int(
                        query_start_loc_cpu[num_spec_decodes].item()
                    )
                if (
                    not _b70_ragged
                    and actual_spec_token_size < expected_spec_token_size
                ):
                    # B70_MTP_PARTIAL_FINAL_GROUP:'''

GDN_ARANGE_OLD = '''                    spec_token_indx = torch.arange(
                        expected_spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
'''

GDN_ARANGE_NEW = '''                    spec_token_indx = torch.arange(
                        actual_spec_token_size
                        if _b70_ragged
                        else expected_spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
'''


def _once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{what} anchor changed or missing; refusing")
    return text.replace(old, new, 1)


def patch_hybrid(text: str) -> str:
    if MARKER in text:
        return text
    text = _once(text, HYBRID_OLD, HYBRID_NEW, "MambaHybridModelState draft classification")
    return _once(text, HYBRID_HELPER_ANCHOR, HYBRID_HELPER + HYBRID_HELPER_ANCHOR.lstrip("\n"), "MambaHybridModelState class")


def patch_gdn(text: str) -> str:
    if MARKER in text:
        return text
    if "B70_MTP_PARTIAL_FINAL_GROUP" not in text or "B70_GDN_SPEC_NO_MASK_SYNC" not in text:
        raise RuntimeError("gdn_attn.py lacks sergiiob/patch_mtp_boundary or b70/patch_gdn_spec_sync; refusing")
    text = _once(text, GDN_FALLBACK_OLD, GDN_FALLBACK_NEW, "GDN zero-draft batch fallback")
    text = _once(text, GDN_PARTIAL_OLD, GDN_PARTIAL_NEW, "GDN partial final group")
    text = _once(text, GDN_ARANGE_OLD, GDN_ARANGE_NEW, "GDN pure-spec token index")
    if "\nimport os\n" not in text:
        text = _once(text, "\nimport torch\n", "\nimport os\n\nimport torch\n", "gdn_attn imports")
    return text


def _apply(path: Path, fn) -> None:
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    original = path.read_text()
    patched = fn(original)
    if patched == original:
        print(f"already patched {path}")
        return
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"patched {path}")


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    root = Path(next(iter(spec.submodule_search_locations)))
    hybrid = root / "v1/worker/gpu/model_states/mamba_hybrid.py"
    gdn = root / "v1/attention/backends/gdn_attn.py"
    # patch both texts before writing either, so a moved anchor leaves the tree untouched
    for p in (hybrid, gdn):
        if not p.is_file():
            raise SystemExit(f"{p} not found")
    patch_hybrid(hybrid.read_text())
    patch_gdn(gdn.read_text())
    _apply(hybrid, patch_hybrid)
    _apply(gdn, patch_gdn)


if __name__ == "__main__":
    main()
