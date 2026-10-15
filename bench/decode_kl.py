#!/usr/bin/env python3
"""Decode-path quality check for GDN state dtype (or anything that only acts during decode).
prompt_logprobs (kl_dump.py) runs the chunked prefill kernel, which keeps the recurrent state on chip and barely touches
the stored state, so it cannot see an fp16 state. Here: generate greedily with logprobs on the arm under test (decode path,
state stored/loaded every token), then score the same token ids with prompt_logprobs on the fp32 reference server and
compare position by position. Run it for the fp32 arm too: decode-vs-prefill numerics differ a little even at fp32, so the
fp32 arm's number is the floor to compare against.

  decode_kl.py gen   PORT MODEL OUT.json [N_TOKENS]   # on the arm under test (and on the fp32 arm)
  decode_kl.py score PORT MODEL GEN.json OUT.json     # on the fp32 reference server, once per GEN file
  decode_kl.py cmp   SCORE.json                       # KL(prefill_fp32 || decode_arm), top-1 agreement, by position bucket
"""
import json, math, os, sys, urllib.request

def post(port, path, body, timeout=900):
    r = urllib.request.Request(f"http://localhost:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

def prompts():
    here = os.path.dirname(os.path.abspath(__file__))
    code = open(os.path.join(here, "corpus.txt")).read()
    prose = open(os.path.join(here, "..", "README.md")).read()
    return [code[200000 + i * 9000: 200000 + i * 9000 + 1500] for i in range(4)] + [prose[i * 4000: i * 4000 + 1500] for i in range(4)]

def tid(k):  # "token_id:123" -> 123
    return int(str(k).split(":")[-1])

mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode == "gen":
    port, model, out = sys.argv[2:5]; n = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    res = []
    for p in prompts():
        ids = post(port, "/tokenize", {"model": model, "prompt": p})["tokens"]
        r = post(port, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": n, "temperature": 0, "logprobs": 20,
                                            "ignore_eos": True, "return_tokens_as_token_ids": True})["choices"][0]["logprobs"]
        gen = [tid(t) for t in r["tokens"]]
        top = [{str(tid(k)): v for k, v in d.items()} for d in r["top_logprobs"]]
        res.append({"prompt_ids": ids, "gen_ids": gen, "top": top})
    json.dump(res, open(out, "w")); print("gen", len(res), "prompts x", n, "tokens ->", out)
elif mode == "score":
    port, model, gen_path, out = sys.argv[2:6]
    res = []
    for g in json.load(open(gen_path)):
        ids = g["prompt_ids"] + g["gen_ids"]
        pl = post(port, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                            "prompt_logprobs": 20})["choices"][0]["prompt_logprobs"]
        P = len(g["prompt_ids"])
        ref = [{k: v["logprob"] for k, v in pl[P + j].items()} for j in range(len(g["gen_ids"]))]
        res.append({"ref": ref, "top": g["top"]})
    json.dump(res, open(out, "w")); print("scored ->", out)
elif mode == "cmp":
    res = json.load(open(sys.argv[2])); buckets = {}
    for r in res:
        for j, (a, b) in enumerate(zip(r["ref"], r["top"])):
            if not a or not b: continue
            S = list(a); floor = min(b.values())
            za = math.log(sum(math.exp(a[k]) for k in S)); zb = math.log(sum(math.exp(b.get(k, floor)) for k in S))
            kl = sum(math.exp(a[k] - za) * ((a[k] - za) - (b.get(k, floor) - zb)) for k in S)
            key = "0-255" if j < 256 else "256-511" if j < 512 else "512+"
            s = buckets.setdefault(key, [0.0, 0, 0]); s[0] += kl; s[1] += max(a, key=a.get) == max(b, key=b.get); s[2] += 1
    for k in ("0-255", "256-511", "512+"):
        if k in buckets:
            kl, ag, n = buckets[k]; print(f"positions {k:8s} KL {kl / n:.4f}  top1 {100 * ag / n:.1f}%  n={n}")
else:
    sys.exit(__doc__)
