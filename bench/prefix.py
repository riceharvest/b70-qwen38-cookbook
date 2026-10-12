import json,sys,time,uuid,urllib.request
BASE=f"http://127.0.0.1:{sys.argv[1]}"; M=json.loads(urllib.request.urlopen(BASE+"/v1/models").read())["data"][0]["id"]
C=open("corpus.txt").read(); base="["+uuid.uuid4().hex+"]\n"+C[500000:560000]
def ttft(p):
    b=json.dumps({"model":M,"prompt":p,"max_tokens":1,"temperature":0}).encode(); t=time.perf_counter()
    u=json.loads(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/completions",b,{"Content-Type":"application/json"})).read())["usage"]
    return time.perf_counter()-t,u["prompt_tokens"]
print("cold  %.2fs %d tok"%ttft(base+"\nQ1: summarize."))
print("warm  %.2fs %d tok (shared prefix, new suffix)"%ttft(base+"\nQ2: list the functions."))
print("warm2 %.2fs %d tok"%ttft(base+"\nQ2: list the functions.\nA: ...\nQ3: and classes?"))
