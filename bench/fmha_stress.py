import torch, sys
sys.path.insert(0,'/opt/b70'); import fmha_bench as fb
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
# exhaustive: every e4m3 byte (minus NaN) as K and V values, compare against torch's own fp8->fp32 conversion
dev='xpu'; HQ,HKV,D,BS=24,4,256,64
allb=torch.arange(256,dtype=torch.uint8); allb=allb[(allb&0x7F)!=0x7F]   # drop NaN encodings
for scale_desc,L in [("all-254-codes",1024),("all-codes-long",8192)]:
    nblk=L//BS
    idx=torch.randint(0,allb.numel(),(nblk,BS,HKV,D))
    kc=allb[idx].view(torch.float8_e4m3fn).to(dev); vc=allb[torch.randint(0,allb.numel(),(nblk,BS,HKV,D))].view(torch.float8_e4m3fn).to(dev)
    bt=torch.arange(nblk,device=dev,dtype=torch.int32).view(1,-1); Tq=256
    q=(torch.randn(Tq,HQ,D,device=dev)*0.02).half(); out=torch.empty_like(q)
    one=torch.ones(1,device=dev); 
    flash_attn_varlen_func(q,kc,vc,Tq,torch.tensor([0,Tq],device=dev,dtype=torch.int32),L,seqused_k=torch.tensor([L],device=dev,dtype=torch.int32),
        causal=True,block_table=bt,softmax_scale=D**-0.5,out=out,fa_version=2,k_descale=one,v_descale=one.clone())
    K=kc.reshape(-1,HKV,D).float().repeat_interleave(6,1); V=vc.reshape(-1,HKV,D).float().repeat_interleave(6,1)
    s=torch.einsum('qhd,khd->hqk',q.float(),K)*D**-0.5; c0=L-Tq
    m=torch.arange(L,device=dev)[None,:]>(c0+torch.arange(Tq,device=dev))[:,None]; s.masked_fill_(m[None],float('-inf'))
    ref=torch.einsum('hqk,khd->qhd',s.softmax(-1),V)
    print(scale_desc, "rel err", ((out.float()-ref).norm()/ref.norm()).item(), "max abs", (out.float()-ref).abs().max().item(), "nan", out.isnan().any().item())
