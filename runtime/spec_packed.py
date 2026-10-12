"""B70: single-pass speculative-verify attention for vllm_xpu_kernels (replaces _spec_decode_varlen_fwd).
All q_len draft positions x GQA heads of a sequence become packed rows of ONE decode query per KV head
(KV read once). Per-row causal limits are applied inside the decode kernel (B70_SPEC_QLEN, patched
paged_decode). Requires libattn_kernels_xe_2 built with the B70 spec mask."""
import os, torch
import vllm_xpu_kernels.flash_attn_interface as fai
_orig = fai._spec_decode_varlen_fwd
_DBG = os.environ.get('B70_SPEC_DEBUG') == '1'; _seen = set()

def _packed(q, k, v, out, cu_seqlens_q, seqused_k, block_table, max_seqlen_k, k_descale, v_descale,
            softmax_scale, s_aux, window_size, softcap):
    B = cu_seqlens_q.numel() - 1; ql = q.shape[0] // B; Hq, D = q.shape[1], q.shape[2]
    Hkv = k.shape[2]; G = Hq // Hkv
    if s_aux is not None or softcap != 0.0 or tuple(window_size) != (-1, -1) or ql < 2:
        return _orig(q, k, v, out, cu_seqlens_q, seqused_k, block_table, max_seqlen_k, k_descale, v_descale,
                     softmax_scale, s_aux, window_size, softcap)
    # [B*ql, Hkv*G, D] -> [B, Hkv*(ql*G), D] rows ordered (kv head, position, head-in-group)
    qp = q.view(B, ql, Hkv, G, D).permute(0, 2, 1, 3, 4).reshape(B, Hkv * ql * G, D)
    cu = torch.arange(B + 1, dtype=torch.int32, device=q.device)
    if _DBG and (B, ql, max_seqlen_k) not in _seen:
        _seen.add((B, ql, max_seqlen_k)); print(f"[B70 spec_packed] B={B} ql={ql} max_seqlen_k={max_seqlen_k} "
              f"capturing={torch.xpu.is_current_stream_capturing() if hasattr(torch.xpu,'is_current_stream_capturing') else '?'}", flush=True)
    os.environ["B70_SPEC_QLEN"] = str(ql)
    try:
        op = fai.flash_attn_varlen_func(qp, k, v, 1, cu, max_seqlen_k, seqused_k=seqused_k, causal=True,
                                        block_table=block_table, softmax_scale=softmax_scale,
                                        fa_version=2, k_descale=k_descale, v_descale=v_descale)
    finally:
        os.environ["B70_SPEC_QLEN"] = "1"
    res = op.view(B, Hkv, ql, G, D).permute(0, 2, 1, 3, 4).reshape(B * ql, Hq, D)
    if out is not None:
        out.copy_(res); return out
    return res

if os.environ.get("B70_SPEC_PACKED", "1") == "1":
    fai._spec_decode_varlen_fwd = _packed
