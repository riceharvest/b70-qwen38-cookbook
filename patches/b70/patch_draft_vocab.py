"""B70: pruned draft vocabulary for the MTP draft LM head (on top of patch_draft_lmhead_int4).
Draft logits are computed only over B70_DRAFT_VOCAB (json list of token ids) and scattered into a
-inf full-vocab tensor. Drafts are greedy (argmax) and verified by the full fp16 target head -> lossless.
Run AFTER patch_draft_lmhead_int4.py. Gate: B70_DRAFT_VOCAB=<path> (unset = unchanged)."""
import pathlib, sys
roots=["/opt/venv/lib/python3.12/site-packages/vllm","/workspace/vllm/vllm"]
MARK="# B70_DRAFT_VOCAB"
OLD_B='''    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(
        weight.detach()
    )'''
NEW_B='''    ''' + MARK + '''
    _vf = os.environ.get("B70_DRAFT_VOCAB")
    _w = weight.detach()
    if _vf:
        import json as _json
        _ids = torch.tensor(sorted(_json.load(open(_vf))), dtype=torch.long, device=_w.device)
        model._b70_draft_vocab_ids = _ids
        model._b70_draft_vocab_full = _w.shape[0]
        _w = _w.index_select(0, _ids).contiguous()
        print(f"[B70] draft vocab: {_ids.numel()} / {weight.shape[0]} rows", flush=True)
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(_w)'''
OLD_L='''    org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", None)'''
NEW_L='''    _ids = getattr(model, "_b70_draft_vocab_ids", None)
    if _ids is not None:  ''' + MARK + '''
        full = logits.new_full((*logits.shape[:-1], model._b70_draft_vocab_full), float("-inf"))
        full.index_copy_(full.dim() - 1, _ids, logits)
        logits = full
    org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", None)'''
n=0
for r in roots:
    p=pathlib.Path(r)/"model_executor/models/b70_draft_lmhead_int4.py"
    if not p.exists(): continue
    s=p.read_text()
    if MARK in s: n+=1; continue
    if s.count(OLD_B)!=1 or s.count(OLD_L)!=1: sys.exit(f"anchor mismatch in {p}")
    p.write_text(s.replace(OLD_B,NEW_B).replace(OLD_L,NEW_L)); n+=1; print("patched",p)
if n==0: sys.exit("helper not found: run patch_draft_lmhead_int4.py first")
