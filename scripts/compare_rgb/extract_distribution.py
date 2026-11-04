#!/usr/bin/env python3
"""Extract standard FID / I3D features from cached, aligned25Hz RGB only."""
import argparse,json,sys,os
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'outputs/rgb_distribution_metrics/python')]
from compare_rgb.common import sha,key,atomic_json,save_npz,ground_truth


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episode-indices');p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--clip-batch-size',type=int,default=2);p.add_argument('--shard-index',type=int,default=int(os.environ.get('RANK',0)));p.add_argument('--num-shards',type=int,default=int(os.environ.get('WORLD_SIZE',1)));p.add_argument('--plan-only',action='store_true');a=p.parse_args()
 if 'LOCAL_RANK' in os.environ:a.device=f'cuda:{os.environ["LOCAL_RANK"]}'
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 panel=json.loads(a.panel.read_text());entries=panel['episodes']
 if a.episode_indices:
  wanted={int(x) for x in a.episode_indices.split(',')};entries=[e for e in entries if e['panel_index'] in wanted]
 plan=dict(episodes=len(entries),common_hz=25,fid_frames=sum((e['num_frames']-1)//2 for e in entries),fvd16_clips=sum(((e['num_frames']-1)//2)//16 for e in entries),fvd32_clips=sum(((e['num_frames']-1)//2)//32 for e in entries),sampling='FID: all scored native25Hz frames; FVD: nonoverlapping complete16/32-frame clips from first predicted frame; no GT frame0, cross-episode clips, interpolation, repetition or tail padding')
 a.output.mkdir(parents=True,exist_ok=True)
 if a.plan_only:atomic_json(plan,a.output/'sample_plan.json');print(json.dumps(plan),flush=True);return
 torch.cuda.set_device(a.device);os.environ.setdefault('TORCH_HOME',str(ROOT/'outputs/pretrained/torch'))
 from pytorch_fid.inception import InceptionV3
 manifest=json.loads((ROOT/'outputs/pretrained/rgb_metrics/manifest.json').read_text());i3dpath=ROOT/'outputs/pretrained/rgb_metrics/i3d_torchscript.pt'
 contract=dict(detectors=manifest,pixel_input='same saved/GT uint8 RGB224x224, no crop',fid='pytorch-fid0.3.0 TF-compatible Inception pool3,2048D; input[0,1], internal299x299 bilinear then[-1,1]',fvd='StyleGAN-V TF-I3D TorchScript Kinetics400; float32 BCTHW input[0,255], rescale=True resize=True return_features=True',clips=[16,32],clip_stride='equals clip length; full native25Hz clip only',TF32=False,compute_dtype='float32',frame_batch_size=a.batch_size,clip_batch_size=a.clip_batch_size)
 fid=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]],resize_input=True,normalize_input=True).to(a.device).eval().requires_grad_(False)
 i3d=torch.jit.load(str(i3dpath),map_location=a.device).eval()
 probe=i3d(torch.zeros(1,3,16,224,224,device=a.device),rescale=True,resize=True,return_features=True)
 assert probe.ndim==2 and probe.shape[0]==1 and torch.isfinite(probe).all()
 i3d_dimension=int(probe.shape[1]);contract['fvd_feature_dimension']=i3d_dimension;contract['torch_version']=torch.__version__
 for e in entries[a.shard_index::a.num_shards]:
  ids=np.arange(2,e['num_frames'],2,dtype=np.int64)
  groups={}
  for m in ['tokenworld','ctrlworld']:
   path=a.artifacts/('tokenworld' if m=='tokenworld' else 'ctrlworld_25hz')/key(e)/'rgb.npz'
   with np.load(path) as data:
    lookup={int(f):i for i,f in enumerate(data['frame_ids'])};assert set(ids)<=lookup.keys();groups[m]=data['rgb'][[lookup[int(f)] for f in ids]]
  groups['real']=ground_truth(e,ids)
  for model in ['real','tokenworld','ctrlworld']:
   directory=a.output/model;directory.mkdir(parents=True,exist_ok=True);meta=directory/f'{key(e)}.json';npz=directory/f'{key(e)}.npz'
   rgb=groups[model];image_hash=__import__('hashlib').sha256(rgb.tobytes()).hexdigest()
   if meta.exists() and npz.exists():
    old=json.loads(meta.read_text());assert old['contract']==contract and old['rgb_sha256']==image_hash
    print('skip features',model,key(e),flush=True);continue
   ff=[]
   for start in range(0,len(rgb),a.batch_size):
    x=torch.from_numpy(rgb[start:start+a.batch_size].copy()).permute(0,3,1,2).to(a.device,dtype=torch.float32)/255
    feature=fid(x)[0].flatten(1);assert feature.shape[1]==2048;ff.append(feature.cpu().numpy())
   arrays=dict(frame_ids=ids,fid=np.concatenate(ff))
   for length in [16,32]:
    starts=np.arange(0,len(rgb)-length+1,length,dtype=np.int64);features=[]
    for start in range(0,len(starts),a.clip_batch_size):
     batch=np.stack([rgb[j:j+length] for j in starts[start:start+a.clip_batch_size]])
     x=torch.from_numpy(batch).permute(0,4,1,2,3).contiguous().to(a.device,dtype=torch.float32)
     v=i3d(x,rescale=True,resize=True,return_features=True);assert v.ndim==2 and v.shape[0]==len(batch)
     features.append(v.cpu().numpy())
    arrays[f'fvd{length}']=np.concatenate(features) if features else np.empty((0,i3d_dimension),dtype=np.float32)
    arrays[f'clip_source_starts{length}']=ids[starts]
   assert all(np.isfinite(v).all() for k,v in arrays.items() if k.startswith(('fid','fvd')))
   save_npz(npz,**arrays);atomic_json(dict(model=model,entry=e,contract=contract,rgb_sha256=image_hash,frame_count=len(ids),fvd16_count=len(arrays['fvd16']),fvd32_count=len(arrays['fvd32']),fvd_feature_dimension=arrays['fvd16'].shape[1],feature_file_sha256=sha(npz)),meta)
   print('features',model,key(e),len(ids),len(arrays['fvd16']),len(arrays['fvd32']),flush=True)
 atomic_json(dict(rank=a.shard_index,world_size=a.num_shards,panel_sha256=sha(a.panel),entries=[e['panel_index'] for e in entries[a.shard_index::a.num_shards]],plan=plan,contract=contract),a.output/f'extraction.rank{a.shard_index:02d}.json')
if __name__=='__main__':main()
