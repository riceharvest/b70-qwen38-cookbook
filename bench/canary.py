import json,sys,urllib.request
BASE=f"http://127.0.0.1:{sys.argv[1] if len(sys.argv)>1 else 8000}"
M=json.loads(urllib.request.urlopen(BASE+"/v1/models").read())["data"][0]["id"]
Q=[("What does `sum(i*i for i in range(4))` evaluate to in Python? Answer with just the number.","14"),
   ("Compute 9*7-3. Answer with just the number.","60"),
   ("What is 17*23? Answer with just the number.","391"),
   ("What does `len([x for x in range(10) if x%3==0])` return in Python? Just the number.","4"),
   ("What is 2**10 - 1000? Just the number.","24"),
   ("What does `sum(range(1,11))` evaluate to? Just the number.","55")]
ok=0
for q,a in Q:
    b=json.dumps({"model":M,"messages":[{"role":"user","content":q}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=json.loads(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",b,{"Content-Type":"application/json"})).read())
    out=(r["choices"][0]["message"]["content"] or "").strip(); good=out.strip("`. ")==a; ok+=good
    print(("OK  " if good else "BAD ")+repr(out)+" expected "+a)
print(f"canary {ok}/{len(Q)}")
