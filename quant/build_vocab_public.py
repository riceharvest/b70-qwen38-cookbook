# Draft-head vocab without serving traffic: technical prose + code + reasoning/agentic calibration text.
# usage: build_vocab_public.py K OUT.json   (container: tokenizer at /model, corpora in /opt/b70)
import json, collections, sys
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/model'); K=int(sys.argv[1]); out=sys.argv[2]
def count_text(s):
    c=collections.Counter()
    for i in range(0,len(s),200000): c.update(tok(s[i:i+200000],add_special_tokens=False)['input_ids'])
    return c
gen=count_text(open('/opt/b70/general.txt',errors='ignore').read()[:60_000_000])
code=count_text(open('/opt/b70/corpus.txt',errors='ignore').read()[:8_000_000])
chat=count_text("\n\n".join(x['text'] for x in json.load(open('/opt/b70/mixedcal.json'))))
def norm(c): T=sum(c.values()); return {t:v/T for t,v in c.items()}
score=collections.Counter()
for c,w in ((norm(chat),1.0),(norm(gen),1.0),(norm(code),0.5)):
    for t,v in c.items(): score[t]+=w*v
sel=set(tok.get_added_vocab().values())|set(tok.all_special_ids)
for t,_ in score.most_common():
    if len(sel)>=K: break
    sel.add(t)
json.dump(sorted(sel),open(out,'w'))
for name,c in (("chat",chat),("general",gen),("code",code)):
    T=sum(c.values()); print(f"K={K} {name} coverage {sum(v for t,v in c.items() if t in sel)/T:.5f}")
