# Mixed-precision quality simulator. usage: qsim.py SPEC_JSON LABEL   (scores vs ref_bf16.pt, appends quality.jsonl)
# SPEC: {"keep_bf16": [regex,...],  # modules taken from BF16 instead of the GPTQ checkpoint
#        "rtn": {"regex": group_size},  # optional: RTN int4 sym of BF16 weights at given group size instead
#        "lm_head": "bf16"|"fp8"|"int4g128", "act8": [regex...] (per-token int8 fake-quant of inputs)}
import torch, json, sys, time, re, math, os
import numpy as np
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
dev='xpu'; SRC='/bf16'; Q='/gptq'; P='model.language_model.'
spec=json.loads(sys.argv[1]); label=sys.argv[2]
keep=[re.compile(r) for r in spec.get("keep_bf16",[])]; rtn={re.compile(k):v for k,v in spec.get("rtn",{}).items()}
act8=[re.compile(r) for r in spec.get("act8",[])]; smooth=[re.compile(r) for r in spec.get("smooth8",[])]; alpha=spec.get("alpha",0.5)
cfg=AutoConfig.from_pretrained(SRC); tc=cfg.text_config; tc._attn_implementation='sdpa'
def opener(root):
    idx=json.load(open(root+'/model.safetensors.index.json'))['weight_map']; H={}
    def get(n):
        f=idx[n]
        if f not in H: H[f]=safe_open(f'{root}/{f}','pt',device='cpu')
        return H[f].get_tensor(n)
    return idx,get
bidx,bget=opener(SRC); qidx,qget=opener(Q)
def unpack(q):
    sh=torch.arange(0,32,4,device=q.device,dtype=torch.int32)
    return ((q.unsqueeze(1)>>sh.view(1,8,1))&0xF).reshape(q.shape[0]*8,q.shape[1])
QCFG=json.load(open(Q+'/config.json')).get('quantization_config',{})
ASYM=QCFG.get('quant_method')=='b70w4'
def gptq_w(prefix):
    qw=qget(prefix+'.qweight').to(dev); sc=qget(prefix+'.scales').to(dev).float(); g=QCFG.get('group_size',128)
    if ASYM:
        qz=qget(prefix+'.qzeros').to(dev); sh=torch.arange(0,32,4,device=dev,dtype=torch.int32)
        z=((qz.unsqueeze(-1)>>sh.view(1,1,8))&0xF).reshape(qz.shape[0],qz.shape[1]*8).float()   # [K/G,N]
        return ((unpack(qw).float()-z.repeat_interleave(g,0))*sc.repeat_interleave(g,0)).t().contiguous().bfloat16()
    return ((unpack(qw).float()-8)*sc.repeat_interleave(g,0)).t().contiguous().bfloat16()   # [N,K]
def rtn_w(w,g,asym=False,mse=False):
    N,K=w.shape; x=w.float().view(N,K//g,g)
    def q(ratio):
        if asym:
            mx=x.amax(-1,keepdim=True)*ratio; mn=x.amin(-1,keepdim=True)*ratio
            s=(mx-mn).clamp(min=1e-8)/15; z=torch.round(-mn/s)
            return (torch.clamp(torch.round(x/s)+z,0,15)-z)*s
        s=x.abs().amax(-1,keepdim=True)*ratio/7.0+1e-12
        return torch.clamp(torch.round(x/s),-8,7)*s
    if not mse: return q(1.0).view(N,K).bfloat16()
    best=q(1.0); berr=((best-x)**2).sum(-1,keepdim=True)
    for r in (0.95,0.9,0.85,0.8,0.75,0.7):
        c=q(r); e=((c-x)**2).sum(-1,keepdim=True); m=e<berr; best=torch.where(m,c,best); berr=torch.where(m,e,berr)
    return best.view(N,K).bfloat16()
R=torch.load(os.environ.get('REF','/opt/b70/ref_bf16.pt')); ids=R["ids"]; N,T=ids.shape; groups=R["groups"]
h=bget(P+'embed_tokens.weight')[ids].to(dev,torch.bfloat16)
with torch.device('meta'): model=M.Qwen3_5TextModel(tc)
rot=M.Qwen3_5TextRotaryEmbedding(config=tc, device=dev)
pos=torch.arange(T,device=dev).view(1,1,-1).expand(4,N,-1); tpos=pos[0]; cos,sin=rot(h,pos[1:])
def a8hook(mod,args):
    x=args[0]; s=x.abs().amax(-1,keepdim=True).float()/127+1e-8
    return (torch.clamp(torch.round(x.float()/s),-127,127)*s).to(x.dtype),
src_count={"bf16":0,"gptq":0,"rtn":0}; t0=time.time(); BS=8
for i,layer in enumerate(model.layers):
    layer.to_empty(device=dev); sd={}
    for k in layer.state_dict().keys():
        full=f'{P}layers.{i}.{k}'; mod=full[:-len('.weight')] if k.endswith('.weight') else None
        if mod and (mod+'.qweight') in qidx:
            rg=[g for r,g in rtn.items() if r.search(mod)]
            if any(r.search(mod) for r in keep): sd[k]=bget(full); src_count["bf16"]+=1
            elif rg: sd[k]=rtn_w(bget(full).to(dev),rg[0],spec.get("asym",False),spec.get("mse",False)); src_count["rtn"]+=1
            else: sd[k]=gptq_w(mod); src_count["gptq"]+=1
        else: sd[k]=qget(full) if (ASYM and full in qidx) else bget(full)
    layer.load_state_dict({k:v for k,v in sd.items()}, strict=True); layer.to(torch.bfloat16)
    for k,v in sd.items():
        p=dict(layer.named_parameters()).get(k)
        if p is not None and v.dtype==torch.float32: p.data=v.to(dev)
    sm=[(n,m) for n,m in layer.named_modules() if isinstance(m,torch.nn.Linear) and any(r.search(f'{P}layers.{i}.{n}') for r in smooth)]
    if sm:
        amax={}
        def stat(name):
            def f(mod,args):
                a=args[0].detach().abs().reshape(-1,args[0].shape[-1]).amax(0).float(); amax[name]=torch.maximum(amax[name],a) if name in amax else a
            return f
        hs=[m.register_forward_pre_hook(stat(n)) for n,m in sm]
        with torch.no_grad():
            for b in range(0,N,BS): layer(h[b:b+BS].clone(), position_embeddings=(cos[b:b+BS],sin[b:b+BS]), attention_mask=None, position_ids=tpos[b:b+BS])
        for hk in hs: hk.remove()
        for n,m in sm:
            full=f'{P}layers.{i}.{n}'; w=bget(full+'.weight').to(dev).float()
            sc=(amax[n].clamp(min=1e-5)**alpha/w.abs().amax(0).clamp(min=1e-5)**(1-alpha)).clamp(min=1e-3)
            m.weight.data=(rtn_w((w*sc).bfloat16(),spec.get("smooth_g",128),spec.get("asym",False),spec.get("mse",False)).float()/sc).bfloat16()
            def mk(sc):
                def f(mod,args):
                    x=args[0]; xs=x.float()/sc; s=xs.abs().amax(-1,keepdim=True)/127+1e-8
                    return ((torch.clamp(torch.round(xs/s),-127,127)*s)*sc).to(x.dtype),
                return f
            m._b70h=m.register_forward_pre_hook(mk(sc))
    hooks=[m.register_forward_pre_hook(a8hook) for n,m in layer.named_modules() if isinstance(m,torch.nn.Linear) and any(r.search(f'{P}layers.{i}.{n}') for r in act8)]
    hooks+=[m._b70h for n,m in sm]
    with torch.no_grad():
        for b in range(0,N,BS):
            h[b:b+BS]=layer(h[b:b+BS], position_embeddings=(cos[b:b+BS],sin[b:b+BS]), attention_mask=None, position_ids=tpos[b:b+BS])
    for hk in hooks: hk.remove()
    layer.to('meta'); torch.xpu.empty_cache()
norm=M.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dev); norm.weight.data=bget(P+'norm.weight').to(dev)
W=bget('lm_head.weight').to(dev); lmh=spec.get("lm_head","bf16")
if lmh=="fp8":
    s=W.float().abs().amax(1,keepdim=True)/448; W=((W.float()/s).to(torch.float8_e4m3fn).float()*s).to(torch.bfloat16)
elif lmh.startswith("int4g"): W=rtn_w(W,int(lmh[5:])).to(torch.bfloat16)
stats={}
with torch.no_grad():
    for n in range(N):
        x=norm(h[n:n+1])[0]; s=stats.setdefault(groups[n],{"d":[],"t":[],"k":[]})
        tid=R["top_ids"][n].long().to(dev); tl=R["top_lp"][n].float().to(dev); rl=R["tok_lp"][n].to(dev)
        for a in range(0,T-1,512):
            e=min(T-1,a+512); lp=torch.log_softmax((x[a:e]@W.t()).float(),-1)
            act=lp.gather(1,ids[n,a+1:e+1].to(dev).view(-1,1)).squeeze(1)
            s["d"].append((rl[a:e]-act).cpu()); s["t"].append((lp.argmax(-1)==tid[a:e,0]).float().cpu())
            pr=tl[a:e].exp(); s["k"].append((pr*(tl[a:e]-lp.gather(1,tid[a:e]))).sum(-1).cpu())
out={"label":label,"spec":spec,"src":src_count,"secs":round(time.time()-t0)}
for g,s in list(stats.items())+[("all",{k:sum((st[k] for st in stats.values()),[]) for k in ("d","t","k")})]:
    d=torch.cat(s["d"]); out[g]={"ppl_ratio":round(math.exp(d.mean().item()),4),"top1_agree":round(torch.cat(s["t"]).mean().item(),4),"kl_top20":round(torch.cat(s["k"]).mean().item(),5)}
print(json.dumps(out)); open('/opt/b70/quality.jsonl','a').write(json.dumps(out)+"\n")
