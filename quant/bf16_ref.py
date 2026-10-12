# Layer-streamed BF16 reference on XPU. Usage: bf16_ref.py OUT.pt [N_per_group]
import torch, json, sys, time
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
dev='xpu'; SRC='/bf16'; P='model.language_model.'; K=20
out_path=sys.argv[1]; nper=int(sys.argv[2]) if len(sys.argv)>2 else 10**9
cfg=AutoConfig.from_pretrained(SRC); tc=cfg.text_config; tc._attn_implementation='sdpa'
idx=json.load(open(SRC+'/model.safetensors.index.json'))['weight_map']; H={}
def get(n):
    f=idx[n]
    if f not in H: H[f]=safe_open(f'{SRC}/{f}','pt',device='cpu')
    return H[f].get_tensor(n)
ev=json.load(open('/opt/b70/eval_tokens.json'))
groups=[]; seqs=[]
for g in tuple(ev):
    for s in ev[g][:nper]: groups.append(g); seqs.append(s)
ids=torch.tensor(seqs); N,T=ids.shape; print("N",N,"T",T,flush=True)
emb=get(P+'embed_tokens.weight')                         # cpu gather
h=emb[ids].to(dev, torch.bfloat16); del emb
with torch.device('meta'): model=M.Qwen3_5TextModel(tc)
rot=M.Qwen3_5TextRotaryEmbedding(config=tc, device=dev)
pos=torch.arange(T,device=dev).view(1,1,-1).expand(4,N,-1); tpos=pos[0]; mpos=pos[1:]
cos,sin=rot(h,mpos)
BS=int(__import__('os').environ.get('BS','8')); t0=time.time()
for i,layer in enumerate(model.layers):
    layer.to_empty(device=dev)
    sd={k: get(f'{P}layers.{i}.{k}') for k in layer.state_dict().keys()}
    layer.load_state_dict(sd, strict=True); layer.to(torch.bfloat16)
    # restore fp32 params that HF keeps fp32 (A_log etc.) from checkpoint dtype
    for k,v in sd.items():
        p=dict(layer.named_parameters()).get(k)
        if p is not None and v.dtype==torch.float32: p.data=v.to(dev)
    with torch.no_grad():
        for b in range(0,N,BS):
            h[b:b+BS]=layer(h[b:b+BS], position_embeddings=(cos[b:b+BS],sin[b:b+BS]), attention_mask=None, position_ids=tpos[b:b+BS])
    layer.to('meta'); torch.xpu.empty_cache()
    if i%8==0 or i==len(model.layers)-1: print(f"layer {i} done {time.time()-t0:.0f}s  h_absmax={h.abs().max().item():.1f}",flush=True)
norm=M.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dev); norm.weight.data=get(P+'norm.weight').to(dev)
W=get('lm_head.weight').to(dev)
tok_lp=torch.empty(N,T-1); top_ids=torch.empty(N,T-1,K,dtype=torch.int32); top_lp=torch.empty(N,T-1,K,dtype=torch.float16)
with torch.no_grad():
    for n in range(N):
        x=norm(h[n:n+1])[0]
        for s in range(0,T-1,512):
            e=min(T-1,s+512); lp=torch.log_softmax((x[s:e]@W.t()).float(),-1)
            tok_lp[n,s:e]=lp.gather(1,ids[n,s+1:e+1].to(dev).view(-1,1)).squeeze(1).cpu()
            v,ix=lp.topk(K,-1); top_ids[n,s:e]=ix.int().cpu(); top_lp[n,s:e]=v.half().cpu()
torch.save({"groups":groups,"ids":ids,"tok_lp":tok_lp,"top_ids":top_ids,"top_lp":top_lp},out_path)
for g in tuple(ev):
    m=torch.tensor([x==g for x in groups])
    if m.any(): print(g,"ppl",torch.exp(-tok_lp[m].mean()).item())
print("total time",time.time()-t0)
