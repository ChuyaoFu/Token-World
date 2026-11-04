import os,sys,zipfile,requests,time,concurrent.futures,shutil
from pathlib import Path
r=Path(os.environ['TOKENWORLD_SIM_STAGE'])
url='https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/3dc3b798668feb99ac61cc9086d84cbcc3d79186/embodiments.zip'
size=219847741
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

import hashlib
p=r/'embodiments.zip'
with p.open('wb') as f:f.truncate(size)
def fetch_block(a):
    b=min(size-1,a+4194304-1)
    with p.open('r+b',buffering=0) as f:f.seek(a);f.write(get(a,b))
with concurrent.futures.ThreadPoolExecutor(12) as pool:
    for n,_ in enumerate(pool.map(fetch_block,range(0,size,4194304))):
        if n%10==0:print(n,flush=True)
assert hashlib.sha256(p.read_bytes()).hexdigest()=='85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789'
dest=r/'sim/assets'
assert not (dest/'embodiments').is_symlink()
with zipfile.ZipFile(p) as z:
    for info in z.infolist():
        if info.filename.startswith('embodiments/aloha-agilex/') and '/._' not in info.filename:
            assert (dest/'embodiments') in (dest/info.filename).resolve().parents
            z.extract(info,dest)
assert (dest/'embodiments/aloha-agilex/meshes/box2_Link.dae').is_file()
print('robot archive verified',flush=True)
