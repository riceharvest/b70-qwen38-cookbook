import base64,json,struct,sys,zlib,urllib.request,time
def png(w,h,px):
    raw=b"".join(b"\x00"+bytes(sum((px(x,y) for x in range(w)),())) for y in range(h))
    ch=lambda t,d:struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d))
    return b"\x89PNG\r\n\x1a\n"+ch(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+ch(b"IDAT",zlib.compress(raw))+ch(b"IEND",b"")
# white image, red square top-left, blue circle bottom-right
def px(x,y):
    if 20<=x<100 and 20<=y<100: return (220,20,20)
    if (x-190)**2+(y-190)**2<45**2: return (20,40,220)
    return (255,255,255)
img=base64.b64encode(png(256,256,px)).decode()
BASE=f"http://127.0.0.1:{sys.argv[1]}"
b=json.dumps({"model":"qwen38","max_tokens":80,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"messages":[{"role":"user","content":[
 {"type":"image_url","image_url":{"url":"data:image/png;base64,"+img}},{"type":"text","text":"Describe the shapes, their colors and positions in one sentence."}]}]}).encode()
t=time.time(); r=json.loads(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",b,{"Content-Type":"application/json"})).read())
print(f"{time.time()-t:.2f}s", r["choices"][0]["message"]["content"])
