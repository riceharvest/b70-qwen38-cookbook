import json,threading,time,urllib.request,uuid,sys
BASE=f"http://127.0.0.1:{sys.argv[1] if len(sys.argv)>1 else 8000}"; C=open("corpus.txt").read()
SYS="You are a precise assistant. Reference material follows.\n"+C[2000000:2016000]
TASKS=[("json","Return a JSON array of 8 objects {id,name,score}. JSON only."),
       ("code","Write a Python function that merges two sorted lists. Code only."),
       ("tool","What's the weather in Berlin?"),
       ("json2","Return JSON {\"cities\": [5 European capitals]} only."),
       ("math","What is 17*23? Just the number."),
       ("code2","Write a bash one-liner counting lines in all .py files.")]
TOOLS=[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
def req(name,msg,out,tag,max_tokens=300,system=SYS):
    body={"model":"qwen38","messages":[{"role":"system","content":system},{"role":"user","content":msg}],"max_tokens":max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    if name=="tool": body["tools"]=TOOLS
    t=time.time()
    try:
        r=json.loads(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900).read())
        m=r["choices"][0]["message"]; out[(tag,name)]=(round(time.time()-t,1),r["choices"][0]["finish_reason"],(m.get("content") or "")[:90].replace("\n"," "),m.get("tool_calls") and m["tool_calls"][0]["function"]["arguments"])
    except Exception as e: out[(tag,name)]=("ERR",str(e)[:200])
out={}
# solo reference
for n,m in TASKS: req(n,m,out,"solo")
long=threading.Thread(target=req,args=("long","Summarize the code above in 3 bullets.",out,"long",100,"["+uuid.uuid4().hex+"]"+C[3000000:3110000]))
long.start(); time.sleep(2)
for tag in ("conc1","conc2"):
    th=[threading.Thread(target=req,args=(n,m,out,tag)) for n,m in TASKS]; [x.start() for x in th]; [x.join() for x in th]
long.join()
for k in sorted(out): print(k, out[k])
same=sum(1 for n,_ in TASKS for tag in ("conc1","conc2") if out.get((tag,n),[None]*3)[2:]==out.get(("solo",n),[None]*3)[2:])
print(f"identical-to-solo: {same}/12")
