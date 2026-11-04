import os,sys,zipfile,requests,time,concurrent.futures,shutil
from pathlib import Path
r=Path(os.environ['TOKENWORLD_SIM_STAGE'])
url='https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/3dc3b798668feb99ac61cc9086d84cbcc3d79186/objects.zip'
size=3737778549
def get(a,b):
    for attempt in range(4):
        try:
            resp=requests.get(url+f'?range_start={a}&end={b}',headers={'Range':f'bytes={a}-{b}'},timeout=(12,75))
            resp.raise_for_status()
            assert resp.status_code==206 and len(resp.content)==b-a+1 and resp.headers['Content-Range'].startswith(f'bytes {a}-{b}/')
            return resp.content
        except Exception:
            if attempt==3:raise
            time.sleep(1)
p=r/'sparse.zip'
with p.open('wb') as f:f.truncate(size);f.seek(3736521413);f.write(get(3736521413,size-1))
classes=set('036_cabinet 056_switch 001_bottle 047_mouse 048_stapler 057_toycar 073_rubikscube 075_bread 077_phone 081_playingcards 112_tea-box 113_coffee-box 107_soap'.split())
with zipfile.ZipFile(p) as z:
    infos=[i for i in z.infolist() if len(i.filename.split('/'))>2 and i.filename.split('/')[0]=='objects' and i.filename.split('/')[1] in classes and '/._' not in i.filename and not i.is_dir()]
    assert set(i.filename.split('/')[1] for i in infos)==classes
    groups=[]
    for i in sorted(infos,key=lambda i:i.header_offset):
        a=i.header_offset;b=min(size-1,a+i.compress_size+4096)
        if groups and a<=groups[-1][1]+1048576 and b-groups[-1][0]<4194304:groups[-1][1]=max(b,groups[-1][1])
        else:groups.append([a,b])
    print(len(infos),'members;',len(groups),'ranges;',sum(b-a+1 for a,b in groups)//1048576,'MiB',flush=True)
    def job(g):
        a,b=g
        with p.open('r+b',buffering=0) as f:f.seek(a);f.write(get(a,b))
    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        for n,_ in enumerate(pool.map(job,groups)):
            if n%20==0:print('ranges',n+1,flush=True)
    for i in infos:
        dest=(r/'sim/assets'/i.filename).resolve()
        assert (r/'sim/assets/objects') in dest.parents
        dest.parent.mkdir(parents=True,exist_ok=True)
        with z.open(i) as src,dest.open('wb') as fd:shutil.copyfileobj(src,fd)
print('asset CRC checks complete',flush=True)
