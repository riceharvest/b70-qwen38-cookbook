# Hash attention outputs (prefill chunk, decode, stock spec verify) for bit-identity checks between kernel builds.
import torch, hashlib, os
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
dev='xpu'; HQ,HKV,D,BS=24,4,256,64; torch.manual_seed(7)
allb=torch.arange(256,dtype=torch.uint8); allb=allb[(allb&0x7F)!=0x7F]
def run(B,Tq,L):
    nblk=L//BS*B
    kc=allb[torch.randint(0,allb.numel(),(nblk,BS,HKV,D))].view(torch.float8_e4m3fn).to(dev)
    vc=allb[torch.randint(0,allb.numel(),(nblk,BS,HKV,D))].view(torch.float8_e4m3fn).to(dev)
    bt=torch.arange(nblk,device=dev,dtype=torch.int32).view(B,-1)
    q=(torch.randn(B*Tq,HQ,D)*0.02).half().to(dev); out=torch.empty_like(q); one=torch.ones(1,device=dev)
    cu=torch.arange(0,B*Tq+1,Tq,device=dev,dtype=torch.int32); sk=torch.full((B,),L,device=dev,dtype=torch.int32)
    flash_attn_varlen_func(q,kc,vc,Tq,cu,L,seqused_k=sk,causal=True,block_table=bt,softmax_scale=D**-0.5,out=out,fa_version=2,k_descale=one*float(os.environ.get("KS","1")),v_descale=one*float(os.environ.get("VS","1")))
    torch.xpu.synchronize(); print(f"B={B} Tq={Tq} L={L}", hashlib.sha1(out.cpu().numpy().tobytes()).hexdigest()[:16])
for B,Tq,L in [(1,256,8192),(1,2048,16384),(4,1,32768),(1,1,65536),(1,5,32768),(8,5,8192)]: run(B,Tq,L)
