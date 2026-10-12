# gen.py PORT L OUT : greedy 200-token answer over the harness ctxdecode context (L target tokens); writes text to OUT
import json, sys, os, urllib.request
port, L, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
C = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")).read()
chars = int(L * 3.0); start = (L * 7919) % (len(C) - chars); ctx = C[start:start + chars]
B = f"http://127.0.0.1:{port}"; M = json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0]["id"]
body = {"model": M, "messages": [{"role": "user", "content": ctx + "\n\nExplain in detail what the code above does, file by file, then propose three concrete refactors with code."}],
        "max_tokens": 200, "temperature": 0, "seed": 1, "chat_template_kwargs": {"enable_thinking": False}}
r = json.loads(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
open(out, "w").write(r["choices"][0]["message"]["content"]); print(out, len(r["choices"][0]["message"]["content"]))
