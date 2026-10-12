# Calibration set without serving traffic: 40 reasoning/agentic (own datasets) + 44 code + 44 technical prose windows, T=2048.
# Code windows stay below byte 8M of corpus.txt and prose below 60M of general.txt (eval windows start there).
import json, random, sys
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/model'); T=2048; random.seed(12345)
mc=[x['text'] for x in json.load(open('/opt/b70/mixedcal.json'))]; mcs=tok("\n\n".join(mc),add_special_tokens=False)['input_ids']
mixed=[mcs[i*T:(i+1)*T] for i in range(min(40,len(mcs)//T))]
def windows(path,n,off,stride,limit):
    txt=open(path,errors='ignore').read(); out=[]
    for i in range(n):
        s=off+i*stride; assert s+30000<=limit; out.append(tok(txt[s:s+30000],add_special_tokens=False)['input_ids'][:T])
    return out
code=windows('/opt/b70/corpus.txt',44,1_000_000,150_000,8_000_000); gen=windows('/opt/b70/general.txt',44,2_000_000,400_000,60_000_000)
ev=json.load(open('/opt/b70/eval_tokens.json')); evset={tuple(s[:64]) for g in ev.values() for s in g}
cal=[s for s in mixed+code+gen if len(s)==T and tuple(s[:64]) not in evset]; random.shuffle(cal)
json.dump(cal,open(sys.argv[1],'w')); print("mixed",len(mixed),"code",len(code),"gen",len(gen),"total",len(cal))
