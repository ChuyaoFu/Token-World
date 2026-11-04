"""Fixed pixel/metric contract for RoboTwin RGB comparison."""
import json,sys,hashlib,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/table1_robotwin')]
import evaluate as table


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def key(e):return f'{e["dataset"]}_{e["episode_index"]:06d}'
def atomic_json(obj,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)
def save_npz(path,**arrays):
 tmp=Path(path).with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,path)


def pixels(rgb):
 """All model RGB is [-1,1]; fixed range, full-frame resize, round-to-nearest."""
 assert rgb.ndim==4 and rgb.shape[1]==3
 x=(rgb.detach().float()+1)/2
 if x.shape[-2:]!=(224,224):x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
 assert torch.isfinite(x).all()
 return x.clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()


def ground_truth(e,frame_ids):
 from decord import VideoReader,cpu
 reader=VideoReader(e['source_video'],ctx=cpu(0),num_threads=2)
 values=torch.from_numpy(reader.get_batch(list(map(int,frame_ids))).asnumpy()).permute(0,3,1,2).float()/127.5-1
 return pixels(values)


def tensor(values,device):return torch.from_numpy(np.asarray(values).copy()).permute(0,3,1,2).float().to(device)/255


class ImageMetrics:
 def __init__(self,device):
  from tokenworld.decoder.metrics.lpips.modules.lpips import LPIPS
  from tokenworld.decoder.metrics.image import ssim
  os.environ.setdefault('TORCH_HOME',str(ROOT/'outputs/pretrained/torch'))
  self.device=device;self.net=LPIPS('alex','0.1').to(device).eval().requires_grad_(False);self.ssim=ssim
 @torch.inference_mode()
 def frame(self,ref,pred):
  ref=tensor(ref[None],self.device);pred=tensor(pred[None],self.device)
  mse=(ref-pred).square().mean().item()
  return dict(mse_01=mse,psnr=120.0 if mse==0 else float(-10*np.log10(mse)),ssim=self.ssim(pred,ref).item(),lpips=self.net(2*pred-1,2*ref-1).item())
 def protocol(self):
  weights=ROOT/'outputs/pretrained/torch/hub/checkpoints'
  return dict(resolution=[224,224],crop='none; full frame',resize='torch bilinear, align_corners=False, antialias=True',pixels='clip [-1,1] -> [0,1] -> round nearest uint8, identical for both models and GT; PSNR/SSIM on [0,1]',lpips='bundled AlexNet ImageNet1K_V1 + LPIPS v0.1; input [-1,1]; one frame per forward',weights={p.name:sha(p) for p in weights.glob('*alex*.pth')},aggregation='primary equal-weight mean of per-frame metrics; also episode and task macro means; PSNR_from_global_MSE separate; context frame0 excluded')
