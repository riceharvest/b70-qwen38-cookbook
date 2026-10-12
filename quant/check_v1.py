# Check a GPTQ v1 export against the raw-zero source, using GPTQModel's v1->v2 step (qzeros += 0x11111111 for 4-bit)
# and its unpack/dequant (right shift by wf, & maxq, scales[g_idx] * (w - z)).
import sys, glob, os, torch
from safetensors import safe_open
raw, v1 = sys.argv[1], sys.argv[2]; wf = torch.arange(0, 32, 4, dtype=torch.int32)
def deq(qw, qz, sc):
    z = ((qz.unsqueeze(2).expand(-1, -1, 8) >> wf.view(1, 1, 8)) & 15).reshape(sc.shape).float()
    w = ((qw.unsqueeze(1).expand(-1, 8, -1) >> wf.view(1, 8, 1)) & 15).reshape(-1, qw.shape[1]).float()
    g = torch.arange(w.shape[0]) // (w.shape[0] // sc.shape[0])
    return sc.float()[g] * (w - z[g])
tot = diff = 0
for f in sorted(glob.glob(raw + '/model-layer*.safetensors')):
    with safe_open(f, 'pt') as a, safe_open(os.path.join(v1, os.path.basename(f)), 'pt') as b:
        for k in [k for k in a.keys() if k.endswith('.qweight')]:
            p = k[:-8]
            wa = deq(a.get_tensor(k), a.get_tensor(p + '.qzeros'), a.get_tensor(p + '.scales'))
            wb = deq(b.get_tensor(k), b.get_tensor(p + '.qzeros') + 0x11111111, b.get_tensor(p + '.scales'))
            d = (wa != wb); tot += wa.numel(); diff += int(d.sum())
            if d.any(): print(p, 'differs in', int(d.sum()), 'elements, max abs', float((wa - wb).abs().max()), 'scale', float(a.get_tensor(p + '.scales').float().abs().max()))
print(f'{diff} of {tot} weights differ')
