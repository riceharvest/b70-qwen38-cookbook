import torch, time, os
import os,sys; sys.path.insert(0,"/opt/b70")
if os.environ.get("PACKED")=="1": import spec_packed  # noqa
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
dev='xpu'; HQ,HKV,D,BS=24,4,256,64; torch.manual_seed(1)
allb=torch.arange(256,dtype=torch.uint8); allb=allb[(allb&0x7F)!=0x7F]
def case(B, qlen, L):
    nblk=(L+BS-1)//BS*B
    kc=allb[torch.randint(0,allb.numel(),(nblk,BS,HKV,D))].view(torch.float8_e4m3fn).to(dev)
    vc=allb[torch.randint(0,allb.numel(),(nblk,BS,HKV,D))].view(torch.float8_e4m3fn).to(dev)
    bt=torch.randperm(nblk,device=dev,dtype=torch.int32).view(B,-1)
    q=(torch.randn(B*qlen,HQ,D,device=dev)*0.02).half(); out=torch.empty_like(q)
    cu=torch.arange(0,B*qlen+1,qlen,device=dev,dtype=torch.int32); sk=torch.full((B,),L,device=dev,dtype=torch.int32)
    one=torch.ones(1,device=dev)
    MK=int(os.environ.get('MAXK',L)); f=lambda: flash_attn_varlen_func(q,kc,vc,qlen,cu,MK,seqused_k=sk,causal=True,block_table=bt,softmax_scale=D**-0.5,out=out,fa_version=2,k_descale=one,v_descale=one.clone())
    for _ in range(10): f()
    ts=[]
    for _ in range(int(os.environ.get("ROUNDS","9"))):
        torch.xpu.synchronize(); t=time.perf_counter()
        for _ in range(30): f()
        torch.xpu.synchronize(); ts.append((time.perf_counter()-t)/30)
    dt=min(ts)
    errs=[]
    for b in range(B):
        K=kc[bt[b]].reshape(-1,HKV,D).float().repeat_interleave(6,1); V=vc[bt[b]].reshape(-1,HKV,D).float().repeat_interleave(6,1)
        qb=q[b*qlen:(b+1)*qlen].float(); s=torch.einsum('qhd,khd->hqk',qb,K)*D**-0.5
        m=torch.arange(L,device=dev)[None,:]>(L-qlen+torch.arange(qlen,device=dev))[:,None]; s.masked_fill_(m[None],float('-inf'))
        ref=torch.einsum('hqk,khd->qhd',s.softmax(-1),V); o=out[b*qlen:(b+1)*qlen].float()
        errs.append(((o-ref).norm()/ref.norm()).item())
    print(f"B={B} qlen={qlen} L={L:6d}  max rel err {max(errs):.3e}  nan={out.isnan().any().item()}  {dt*1e6:8.1f} us/layer")
import os
CASES=[tuple(map(int,c.split("x"))) for c in os.environ.get("CASES","1x1x4096,4x1x32768,1x5x4096,1x5x65536,4x5x16384,1x16x8192").split(",")]
for B,qlen,L in CASES: case(B,qlen,L)
