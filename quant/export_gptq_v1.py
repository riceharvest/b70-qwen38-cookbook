# Export a b70quant checkpoint (raw zero-points) to standard GPTQ v1 (qzeros store z-1), loadable by
# AutoGPTQ/GPTQModel/transformers/vLLM. z=0 never fits v1; such groups get z=1, q=min(q+1,15) (prints how many).
# usage: export_gptq_v1.py SRC_DIR DST_DIR    (DST gets rewritten layer files + hardlinks/copies of the rest)
import sys, os, json, glob, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
SH = torch.arange(0, 32, 4, dtype=torch.int32)
def unpack_n(z):   # int32 [R, C/8] packed along last dim -> int32 [R, C]
    return ((z.unsqueeze(-1) >> SH) & 0xF).reshape(z.shape[0], -1)
def pack_n(u):     # int32 [R, C] (0..15) -> int32 [R, C/8]
    return (u.view(u.shape[0], -1, 8) << SH).sum(-1).to(torch.int32)
fixed = 0
for f in sorted(os.listdir(src)):
    p = os.path.join(src, f)
    if f.endswith('.safetensors'):
        with safe_open(p, 'pt') as s:
            keys = list(s.keys())
            if not any(k.endswith('.qzeros') for k in keys):
                shutil.copy2(p, os.path.join(dst, f)); continue
            T = {k: s.get_tensor(k) for k in keys}
        for k in [k for k in keys if k.endswith('.qzeros')]:
            base = k[:-len('.qzeros')]; z = unpack_n(T[k])          # [K/G, N]
            bad = (z == 0).nonzero()
            if len(bad):
                qw = T[base + '.qweight']                            # [K/8, N] packed along K
                q = ((qw.unsqueeze(1) >> SH.view(1, 8, 1)) & 0xF).reshape(-1, qw.shape[1])   # [K, N]
                G = q.shape[0] // z.shape[0]
                for g, n in bad.tolist():
                    q[g*G:(g+1)*G, n] = (q[g*G:(g+1)*G, n] + 1).clamp(max=15); z[g, n] = 1; fixed += 1
                T[base + '.qweight'] = (q.view(-1, 8, q.shape[1]) << SH.view(1, 8, 1)).sum(1).to(torch.int32)
            T[k] = pack_n(z - 1)
        save_file(T, os.path.join(dst, f), metadata={'format': 'pt'}); print('rewrote', f, flush=True)
    elif f in ('quantize_config.json', 'smooth_log.json'):
        continue
    elif os.path.isfile(p):
        shutil.copy2(p, os.path.join(dst, f))
qc = json.load(open(os.path.join(src, 'quantize_config.json')))
qc['meta'] = {'quantizer': ['b70quant: GPTQ asym int4 g128 + MSE clip search'], 'zero_format': 'gptq v1 (stored z-1)'}
qc['checkpoint_format'] = 'gptq'
json.dump(qc, open(os.path.join(dst, 'quantize_config.json'), 'w'), indent=2)
print('z=0 groups fixed:', fixed)
