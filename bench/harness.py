#!/usr/bin/env python3
"""B70 A/B harness: prefill sweep (unique-nonce, cache-defeating) + decode suite with MTP acceptance deltas.
usage: harness.py --port 8000 --label X [--prefill 1000,8000,32000] [--decode] [--reps 2]"""
import argparse, json, re, time, uuid, urllib.request, os
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000); ap.add_argument("--label", required=True)
ap.add_argument("--prefill", default="1000,8000,32000,64000"); ap.add_argument("--decode", action="store_true")
ap.add_argument("--reps", type=int, default=2); ap.add_argument("--ctxdecode", default="")
ap.add_argument("--sample", action="store_true", help="traffic sampling: T=1.0 top_p=0.95 top_k=20 seed=1"); ap.add_argument("--out", default=os.path.expanduser("~/b70_prep/opt/results.jsonl"))
a = ap.parse_args(); BASE = f"http://127.0.0.1:{a.port}"
MODEL = json.loads(urllib.request.urlopen(BASE + "/v1/models").read())["data"][0]["id"]
CORPUS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")).read()

def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics").read().decode()
    m = {}
    for k in ["spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total", "num_requests_running"]:
        r = re.search(rf"^vllm:{k}{{[^}}]*}} ([0-9.e+]+)", txt, re.M); m[k] = float(r.group(1)) if r else 0.0
    for p in re.finditer(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total{[^}]*position="(\d)"[^}]*} ([0-9.e+]+)', txt, re.M):
        m[f"pos{p.group(1)}"] = float(p.group(2))
    return m

def wait_idle():
    for _ in range(600):
        if metrics()["num_requests_running"] == 0: return
        time.sleep(0.5)

def stream(path, body):
    body = dict(body, model=MODEL, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch:
                c = ch[0]; txt = c.get("text") or (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content") or (c.get("delta") or {}).get("reasoning")
                if txt and first is None: first = time.perf_counter()
    return t0, first, time.perf_counter(), usage

def emit(rec):
    rec.update(label=a.label, ts=time.strftime("%F %T")); print(json.dumps(rec), flush=True)
    open(a.out, "a").write(json.dumps(rec) + "\n")

DECODE = [
  ("code", False, "Write a complete Python module implementing a thread-safe LRU cache with per-key TTL, plus pytest tests. Code only."),
  ("json", False, "Return a JSON array of 25 objects describing fictional employees with fields id, name, department, salary, start_date, skills (array). JSON only."),
  ("prose", False, "Write a vivid 500-word short story about a lighthouse keeper who discovers the light has been signalling to someone."),
  ("think", True, "A train leaves at 14:05 going 80 km/h; another leaves the same station at 14:35 going 110 km/h on a parallel track. When and where does the second catch up? Explain."),
]

SAMP = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1} if a.sample else {"temperature": 0, "seed": 1}

def decode_rec(kind, extra, t0, first, end, u, m0, m1):
    ct = u["completion_tokens"]; d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    drafts = d.get("spec_decode_num_drafts_total", 0)
    emit(dict(kind=kind, **extra, sample=a.sample, ptoks=u["prompt_tokens"], ctoks=ct, decode_tps=round((ct - 1) / (end - first), 1),
              ttft=round(first - t0, 3), tok_per_step=round(ct / drafts, 2) if drafts else None,
              accept_pos=[round(d.get(f"pos{i}", 0) / drafts, 2) for i in range(8) if f"pos{i}" in d] if drafts else None))

# long-context decode: fixed (cacheable) code context, then a continuation task; rep 0 pays prefill, later reps hit the prefix cache
for L in [int(x) for x in a.ctxdecode.split(",") if x]:
    chars = int(L * 3.0); start = (L * 7919) % (len(CORPUS) - chars)
    ctx = CORPUS[start:start + chars]
    for rep in range(a.reps):
        wait_idle(); m0 = metrics()
        t0, first, end, u = stream("/v1/chat/completions", {"messages": [{"role": "user", "content": ctx + "\n\nExplain in detail what the code above does, file by file, then propose three concrete refactors with code."}],
            "max_tokens": 512, "ignore_eos": True, **dict(SAMP, seed=SAMP["seed"] + rep), "chat_template_kwargs": {"enable_thinking": False}})
        decode_rec("ctxdecode", {"target": L, "rep": rep}, t0, first, end, u, m0, metrics())

if a.decode:
    for rep in range(a.reps):
        for name, think, prompt in DECODE:
            wait_idle(); m0 = metrics()
            t0, first, end, u = stream("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": 512,
                **dict(SAMP, seed=SAMP["seed"] + rep), "chat_template_kwargs": {"enable_thinking": think}})
            m1 = metrics(); ct = u["completion_tokens"]
            d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
            drafts = d.get("spec_decode_num_drafts_total", 0)
            emit(dict(kind="decode", case=name, rep=rep, sample=a.sample, ctoks=ct, decode_tps=round((ct - 1) / (end - first), 1),
                      ttft=round(first - t0, 3), tok_per_step=round(ct / drafts, 2) if drafts else None,
                      accept_pos=[round(d.get(f"pos{i}", 0) / drafts, 2) for i in range(8) if f"pos{i}" in d] if drafts else None))
for L in [int(x) for x in a.prefill.split(",") if x]:
    for rep in range(1 if L >= 64000 else a.reps):
        chars = int(L * 3.0); start = (L * 7919 + rep * 104729) % (len(CORPUS) - chars)
        p = "[" + uuid.uuid4().hex + "]\n" + CORPUS[start:start + chars] + "\n\nSummarize the above code."
        wait_idle()
        t0, first, end, u = stream("/v1/completions", {"prompt": p, "max_tokens": 64, "temperature": 0, "ignore_eos": True})
        pt = u["prompt_tokens"]
        emit(dict(kind="prefill", target=L, ptoks=pt, ttft=round(first - t0, 2), prefill_tps=round(pt / (first - t0)),
                  decode_tps_at_ctx=round((u["completion_tokens"] - 1) / (end - first), 1)))
