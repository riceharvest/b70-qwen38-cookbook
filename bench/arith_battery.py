# 100 seeded in-head Python/arithmetic questions (the canary's failure class), greedy, thinking off.
# usage: arith_battery.py [PORT] [N]  -> prints score; per-question results to stdout with -v
import json, random, sys, urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else 8000; N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
BASE = f"http://127.0.0.1:{PORT}"; M = json.loads(urllib.request.urlopen(BASE + "/v1/models").read())["data"][0]["id"]
rng = random.Random(1234)
def gen():
    k = rng.randrange(8); a, b, c = rng.randint(2, 9), rng.randint(2, 13), rng.randint(2, 30)
    if k == 0: e = f"sum(i*i for i in range({a}))"
    elif k == 1: e = f"len([x for x in range({c}) if x%{a}==0])"
    elif k == 2: e = f"{a}*{b}-{c}"
    elif k == 3: e = f"sum(range({a},{c}+1))"
    elif k == 4: e = f"2**{b} - {c*10}"
    elif k == 5: e = f"{c*b} // {a} + {c} % {a}"
    elif k == 6: e = f"max([{a},{b},{c}]) - min([{a},{b},{c}])"
    else: e = f"sum(x for x in range({c}) if x%2==1)"
    return e, str(eval(e))
Q = [gen() for _ in range(N)]
ok = 0
for e, ans in Q:
    q = f"What does `{e}` evaluate to in Python? Answer with just the number."
    b = json.dumps({"model": M, "messages": [{"role": "user", "content": q}], "max_tokens": 16, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", b, {"Content-Type": "application/json"})).read())
    out = (r["choices"][0]["message"]["content"] or "").strip().strip("`. "); good = out == ans; ok += good
    if "-v" in sys.argv: print(("OK  " if good else "BAD ") + e + " -> " + repr(out) + " expected " + ans)
print(f"arith battery {ok}/{N}")
