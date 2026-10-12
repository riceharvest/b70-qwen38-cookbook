# B70 GPTQ quantizer: asym int4 g128 + MSE-grid scales (+ optional SmoothQuant folding), layer-streamed on XPU.
# usage: b70quant.py OUT_DIR [--smooth ALPHA] [--calib calib_tokens.json] [--nseq 128]
import torch, json, sys, time, re, os, math, argparse, shutil, glob
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
ap=argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--smooth',type=float,default=0.0)
ap.add_argument('--calib',default='/opt/b70/calib_tokens.json'); ap.add_argument('--nseq',type=int,default=128)
ap.add_argument('--group',type=int,default=128); ap.add_argument('--damp',type=float,default=0.01)
torch.set_grad_enabled(False)
a=ap.parse_args(); dev='xpu'; SRC='/bf16'; P='model.language_model.'; G=a.group
QMODS=re.compile(r'\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)$')
cfg=AutoConfig.from_pretrained(SRC); tc=cfg.text_config; tc._attn_implementation='sdpa'
idx=json.load(open(SRC+'/model.safetensors.index.json'))['weight_map']; H_={}
def get(n):
    f=idx[n]
    if f not in H_: H_[f]=safe_open(f'{SRC}/{f}','pt',device='cpu')
    return H_[f].get_tensor(n)
os.makedirs(a.out,exist_ok=True)
# ---------------- quant primitives ----------------
GRID=(1.0,0.97,0.94,0.91,0.88,0.85,0.82,0.79,0.76,0.73,0.70)
def find_params(w):            # w [N,G] fp32 -> scale,zero [N,1]  (asym, MSE grid on clip ratio, err^2.4)
    xmin=w.amin(1,keepdim=True).clamp(max=0); xmax=w.amax(1,keepdim=True).clamp(min=0)
    best=None
    for r in GRID:
        mn,mx=xmin*r,xmax*r; s=((mx-mn)/15).clamp(min=1e-8); z=torch.round(-mn/s)
        q=(torch.clamp(torch.round(w/s)+z,0,15)-z)*s; e=(q-w).abs().pow(2.4).sum(1,keepdim=True)
        if best is None: best=(e,s,z)
        else:
            m=e<best[0]; best=(torch.where(m,e,best[0]),torch.where(m,s,best[1]),torch.where(m,z,best[2]))
    return best[1],best[2]
def gptq(W,H):                 # W [N,K] fp32, H [K,K] fp32 -> q uint [N,K], scales [N,K/G], zeros [N,K/G]
    N,K=W.shape; W=W.clone(); H=H.clone()
    dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
    damp=a.damp; md=torch.mean(torch.diag(H)); idx=torch.arange(K,device=dev)
    while True:
        Hd=H.clone(); Hd[idx,idx]+=damp*md
        try:
            L=torch.linalg.cholesky(Hd); Hinv=torch.cholesky_inverse(L); Hinv=torch.linalg.cholesky(Hinv,upper=True); break
        except Exception:
            damp*=10; print(f"    cholesky retry damp={damp}",flush=True)
    del Hd, L, H
    Q=torch.zeros(N,K,dtype=torch.uint8,device=dev); S=torch.zeros(N,K//G,device=dev); Z=torch.zeros(N,K//G,device=dev)
    for i1 in range(0,K,G):
        i2=i1+G; W1=W[:,i1:i2].clone(); E1=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
        s,z=find_params(W1); S[:,i1//G]=s[:,0]; Z[:,i1//G]=z[:,0]
        for i in range(G):
            w=W1[:,i]; d=Hi[i,i]
            q=torch.clamp(torch.round(w/s[:,0])+z[:,0],0,15); Q[:,i1+i]=q.to(torch.uint8)
            dq=(q-z[:,0])*s[:,0]; err=(w-dq)/d
            W1[:,i:]-=err.unsqueeze(1)*Hi[i,i:].unsqueeze(0); E1[:,i]=err
        W[:,i2:]-=E1@Hinv[i1:i2,i2:]
    return Q,S,Z
def dequant(Q,S,Z): return (Q.float()-Z.repeat_interleave(G,1))*S.repeat_interleave(G,1)
def pack(Q,S,Z):               # -> qweight int32 [K/8,N] (LSB-first along K), scales fp16 [K/G,N], qzeros int32 [K/G,N/8] (along N)
    N,K=Q.shape; sh=torch.arange(0,32,4,device=dev,dtype=torch.int32)
    qt=Q.t().to(torch.int32).contiguous()                                     # [K,N]
    qw=(qt.view(K//8,8,N)<<sh.view(1,8,1)).sum(1).to(torch.int32)
    zt=Z.t().to(torch.int32).contiguous()                                     # [K/G,N]
    qz=(zt.view(K//G,N//8,8)<<sh.view(1,1,8)).sum(-1).to(torch.int32)
    return qw.cpu(),S.t().contiguous().half().cpu(),qz.cpu()
# ---------------- calibration ----------------
cal=json.load(open(a.calib))[:a.nseq]; ids=torch.tensor(cal); N_,T=ids.shape
h=get(P+'embed_tokens.weight')[ids].to(dev,torch.bfloat16)
with torch.device('meta'): model=M.Qwen3_5TextModel(tc)
rot=M.Qwen3_5TextRotaryEmbedding(config=tc, device=dev)
pos=torch.arange(T,device=dev).view(1,1,-1).expand(4,N_,-1); tpos=pos[0]; cos,sin=rot(h,pos[1:])
BS=8; out={}; t0=time.time(); smooth_log={}; wm={}
def save_shard(name,tensors):
    save_file(tensors,f'{a.out}/{name}',metadata={'format':'pt'})
    for n in tensors: wm[n]=name
NORM_GROUPS=[('input_layernorm',re.compile(r'(self_attn\.(q|k|v)_proj|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b))$')),
             ('post_attention_layernorm',re.compile(r'mlp\.(gate|up)_proj$'))]
def run(layer):
    with torch.no_grad():
        for b in range(0,N_,BS):
            h[b:b+BS]=layer(h[b:b+BS], position_embeddings=(cos[b:b+BS],sin[b:b+BS]), attention_mask=None, position_ids=tpos[b:b+BS])
def run_stats(layer, mods):
    Hs={}; amax={}; cnt=[0]
    def mk(n):
        def f(m,args):
            x=args[0].reshape(-1,args[0].shape[-1]).float()
            Hs[n]=Hs.get(n,0)+x.t()@x; am=x.abs().amax(0); amax[n]=torch.maximum(amax[n],am) if n in amax else am
        return f
    hk=[m.register_forward_pre_hook(mk(n)) for n,m in mods]
    with torch.no_grad():
        for b in range(0,N_,BS): layer(h[b:b+BS].clone(), position_embeddings=(cos[b:b+BS],sin[b:b+BS]), attention_mask=None, position_ids=tpos[b:b+BS])
    for x in hk: x.remove()
    return Hs,amax
MAXL=int(os.environ.get('MAXL','999'))
def mem(tag): os.environ.get('QDBG') and print(f'  [{tag}] alloc={torch.xpu.memory_allocated()/2**30:.2f}G',flush=True)
for i,layer in enumerate(model.layers):
    if i>=MAXL: break
    mem('start'); layer.to_empty(device=dev); sd={k:get(f'{P}layers.{i}.{k}') for k in layer.state_dict().keys()}
    layer.load_state_dict(sd,strict=True); layer.to(torch.bfloat16)
    for k,v in sd.items():
        p=dict(layer.named_parameters()).get(k)
        if p is not None and v.dtype==torch.float32: p.data=v.to(dev)
    lin={n:m for n,m in layer.named_modules() if isinstance(m,torch.nn.Linear)}
    qmods=[(n,m) for n,m in lin.items() if QMODS.search('.'+n)]
    statmods=[(n,m) for n,m in lin.items() if QMODS.search('.'+n) or (a.smooth>0 and re.search(r'in_proj_(a|b)$',n))]
    Hs,amax=run_stats(layer,statmods); mem('stats')
    if a.smooth>0:      # fold per-channel smoothing into the feeding RMSNorm (Qwen3_5RMSNorm: x_n*(1+w))
        for normname,rx in NORM_GROUPS:
            cons=[n for n in lin if rx.search(n)]
            if not cons: continue
            ax=amax[cons[0]]; aw=torch.stack([lin[n].weight.float().abs().amax(0) for n in cons]).amax(0)
            s=(ax.clamp(min=1e-5)**a.smooth/aw.clamp(min=1e-5)**(1-a.smooth)).clamp(1e-2,1e2)
            norm=dict(layer.named_modules())[normname]
            norm.weight.data=((1+norm.weight.float())/s-1).to(norm.weight.dtype)
            for n in cons:
                lin[n].weight.data=(lin[n].weight.float()*s.unsqueeze(0)).to(lin[n].weight.dtype)
                if n in Hs: Hs[n]=Hs[n]/(s.unsqueeze(0)*s.unsqueeze(1))
            smooth_log[f'{i}.{normname}']=[round(s.min().item(),3),round(s.max().item(),3)]
    for n,m in qmods:
        Q,S,Z=gptq(m.weight.float(),Hs[n]); m.weight.data=dequant(Q,S,Z).to(m.weight.dtype)
        qw,sc,qz=pack(Q,S,Z); base=f'{P}layers.{i}.{n}'
        out[base+'.qweight']=qw; out[base+'.scales']=sc; out[base+'.qzeros']=qz
    del Hs; mem('gptq')
    for k,v in layer.state_dict().items():           # unquantized tensors of this layer (incl. smoothed norms / a,b proj)
        full=f'{P}layers.{i}.{k}'; mod=full[:-7] if k.endswith('.weight') else None
        if not (mod and QMODS.search(mod)): out[full]=v.detach().to(sd[k].dtype).cpu().contiguous()
    save_shard(f'model-layer{i:02d}.safetensors',out); out={}
    run(layer); mem('run'); layer.to('meta'); torch.xpu.empty_cache(); mem('meta')
    if os.environ.get('QDBG'):
        import gc; gc.collect(); torch.xpu.empty_cache(); mem('gc')
        big=[(tuple(o.shape),o.dtype) for o in gc.get_objects() if torch.is_tensor(o) and o.device.type=='xpu' and o.numel()*o.element_size()>50e6]
        print('  live big xpu tensors:',big[:20],flush=True)
        for o in gc.get_objects():
            if torch.is_tensor(o) and o.device.type=='xpu' and o.dim()==2 and o.shape[0]==17408 and o.shape[1]==17408:
                for r in gc.get_referrers(o):
                    print('   ref by',type(r).__name__, (list(r.keys())[:6] if isinstance(r,dict) else str(r)[:120]),flush=True)
                    for rr in gc.get_referrers(r)[:4]: print('      <-',type(rr).__name__, str(rr)[:150] if not isinstance(rr,dict) else list(rr.keys())[:8])
                break
    print(f"layer {i} done {time.time()-t0:.0f}s",flush=True)
# untouched tensors (embed, final norm, lm_head, mtp, visual) straight from BF16, in ~4GB shards
rest=[n for n in idx if not n.startswith(P+'layers.')]; part={}; sz=0; k=0
for n in sorted(rest):
    t=get(n).contiguous(); part[n]=t; sz+=t.numel()*t.element_size()
    if sz>4e9: save_shard(f'model-extra{k}.safetensors',part); part={}; sz=0; k+=1
if part: save_shard(f'model-extra{k}.safetensors',part)
json.dump({"metadata":{},"weight_map":wm},open(f'{a.out}/model.safetensors.index.json','w'),indent=1)
for f in glob.glob(SRC+'/*'):
    b=os.path.basename(f)
    if not b.endswith('.safetensors') and b!='model.safetensors.index.json': shutil.copy2(f,f'{a.out}/{b}')
c=json.load(open(f'{a.out}/config.json'))
c['quantization_config']={"quant_method":"b70w4","bits":4,"group_size":G,"sym":False,"zero_layout":"K/G,N/8 packed-along-N",
  "smooth_alpha":a.smooth,"a8_ok":"input_layernorm+post_attention_layernorm consumers" if a.smooth>0 else None,
  "modules_not_quantized":["lm_head","mtp","visual","in_proj_a","in_proj_b"]}
json.dump(c,open(f'{a.out}/config.json','w'),indent=1); json.dump(smooth_log,open(f'{a.out}/smooth_log.json','w'))
print("DONE",time.time()-t0)
