# Replay serving-shaped prefill attention: q chunk (Tq) attends causally to paged KV of length L (q at the tail).
import torch, time, sys, os
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
torch.manual_seed(0); dev='xpu'
HQ, HKV, D, BS = 24, 4, 256, 64
def run(Tq, L, kv_dtype=torch.float8_e4m3fn, iters=5, check=False):
    nblk = (L + BS - 1)//BS
    kc = (torch.randn(nblk+8, BS, HKV, D, device=dev)*0.5).to(kv_dtype)
    vc = (torch.randn(nblk+8, BS, HKV, D, device=dev)*0.5).to(kv_dtype)
    bt = torch.randperm(nblk+8, device=dev, dtype=torch.int32)[:nblk].view(1, -1)
    q = torch.randn(Tq, HQ, D, device=dev, dtype=torch.float16)
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, Tq], device=dev, dtype=torch.int32); seqk = torch.tensor([L], device=dev, dtype=torch.int32)
    kw = {}
    if kv_dtype == torch.float8_e4m3fn:
        one = torch.ones(1, device=dev, dtype=torch.float32); kw = dict(k_descale=one, v_descale=one.clone())
    f = lambda: flash_attn_varlen_func(q, kc, vc, Tq, cu_q, L, seqused_k=seqk, causal=True, block_table=bt,
                                       softmax_scale=D**-0.5, out=out, fa_version=2, **kw)
    f(); torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(iters): f()
    torch.xpu.synchronize(); dt = (time.perf_counter()-t)/iters
    ctx0 = L - Tq  # causal: query i sees ctx0+i+1 keys
    flops = 4*HQ*D*(Tq*ctx0 + Tq*(Tq+1)/2)
    err = None
    if check:
        K = kc[bt[0]].reshape(-1, HKV, D)[:L].to(torch.float16).float(); V = vc[bt[0]].reshape(-1, HKV, D)[:L].to(torch.float16).float()
        K = K.repeat_interleave(HQ//HKV, 1); V = V.repeat_interleave(HQ//HKV, 1)
        s = torch.einsum('qhd,khd->hqk', q.float(), K) * D**-0.5
        mask = torch.arange(L, device=dev)[None, :] > (ctx0 + torch.arange(Tq, device=dev))[:, None]
        s.masked_fill_(mask[None], float('-inf')); ref = torch.einsum('hqk,khd->qhd', s.softmax(-1), V)
        err = ((out.float()-ref).norm()/ref.norm()).item()
    return dt, flops/dt/1e12, err
if __name__ == "__main__":
    print(run(512, 2048, check=True)[2], "rel err (fp8 kv) small check")
    print(run(512, 2048, kv_dtype=torch.float16, check=True)[2], "rel err (fp16 kv) small check")
    for L in [8192, 16384, 32768, 65536, 131072, 262144]:
        for kvd in (torch.float8_e4m3fn, torch.float16):
            dt, tf, _ = run(8192, L, kv_dtype=kvd, iters=3 if L > 65536 else 5)
            print(f"Tq=8192 L={L:6d} kv={str(kvd)[6:]:14s} {dt*1e3:8.1f} ms/layer  {tf:6.1f} TFLOPS  -> x16 layers {dt*16:.2f}s", flush=True)
