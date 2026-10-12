import json,sys,time,urllib.request
BASE=f"http://127.0.0.1:{sys.argv[1]}"; target=int(sys.argv[2])
C=open("corpus.txt").read(); chars=int(target*3.9)
hay=C[:chars]; mid=len(hay)*3//5
hay=hay[:mid]+"\n# NOTE: The secret passphrase is MAGENTA-OTTER-42.\n"+hay[mid:]
msg=hay+"\n\nWhat is the secret passphrase mentioned in a NOTE comment above? Answer with just the passphrase."
b=json.dumps({"model":"qwen38","messages":[{"role":"user","content":msg}],"max_tokens":30,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
t=time.time(); r=json.loads(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",b,{"Content-Type":"application/json"}),timeout=3600).read())
dt=time.time()-t; u=r["usage"]
print(f"prompt_tokens={u['prompt_tokens']} time={dt:.0f}s ({u['prompt_tokens']/dt:.0f} tok/s) answer={r['choices'][0]['message']['content']!r}")
