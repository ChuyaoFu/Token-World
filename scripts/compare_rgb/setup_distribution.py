#!/usr/bin/env python3
"""Install isolated FID code and cache the authors' FID/I3D detector weights."""
import json,sys,os,hashlib,subprocess,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
URLS={
 'pt_inception-2015-12-05-6726825d.pth':'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth',
 'i3d_torchscript.pt':'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1',
}

EXPECTED={'pt_inception-2015-12-05-6726825d.pth':'6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2','i3d_torchscript.pt':'bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2'}

def main():
 runtime=ROOT/'outputs/rgb_distribution_metrics/python';weights=ROOT/'outputs/pretrained/rgb_metrics';weights.mkdir(parents=True,exist_ok=True)
 if not (runtime/'pytorch_fid').is_dir():subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(runtime),'pytorch-fid==0.3.0'],check=True)
 results=[]
 for name,url in URLS.items():
  path=weights/name
  if not path.exists():
   temp=path.with_suffix('.download')
   with urllib.request.urlopen(url,timeout=45) as response,temp.open('wb') as f:
    while True:
     block=response.read(1024*1024)
     if not block:break
     f.write(block)
   assert temp.stat().st_size>1000000,(name,'invalid model download')
   temp.replace(path)
  digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==EXPECTED[name],f'Incomplete or unexpected detector weights: {path}'
  results.append(dict(name=name,url=url,sha256=digest,bytes=path.stat().st_size))
 # pytorch-fid uses torch.hub's standard weights cache; point it to the same verified file.
 hub=Path(os.environ.get('TORCH_HOME',ROOT/'outputs/pretrained/torch'))/'hub/checkpoints';hub.mkdir(parents=True,exist_ok=True)
 source=weights/'pt_inception-2015-12-05-6726825d.pth';dest=hub/source.name
 if not dest.exists():
  try:os.link(source,dest)
  except OSError:dest.symlink_to(source)
 else:assert hashlib.sha256(dest.read_bytes()).hexdigest()==results[0]['sha256']
 (weights/'manifest.json').write_text(json.dumps(dict(pytorch_fid_version='0.3.0',detectors=results),indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
