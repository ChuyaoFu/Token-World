#!/usr/bin/env python3
"""Compute paired RGB metrics only from saved, source-indexed trajectory artifacts."""
import argparse,json,sys,os,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_rgb.common import key,sha,atomic_json,ground_truth,ImageMetrics


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--grid',type=int,choices=[25,50],default=25);p.add_argument('--device',default='cuda:0');p.add_argument('--episode-indices');p.add_argument('--max-source-frames',type=int,default=0);p.add_argument('--shard-index',type=int,default=int(os.environ.get('RANK',0)));p.add_argument('--num-shards',type=int,default=int(os.environ.get('WORLD_SIZE',1)));a=p.parse_args()
 if 'LOCAL_RANK' in os.environ:a.device=f'cuda:{os.environ["LOCAL_RANK"]}'
 torch.set_num_threads(4);torch.cuda.set_device(a.device);metric=ImageMetrics(a.device);panel=json.loads(a.panel.read_text());entries=panel['episodes']
 if a.episode_indices:
  wanted={int(x) for x in a.episode_indices.split(',')};entries=[e for e in entries if e['panel_index'] in wanted]
 entries=entries[a.shard_index::a.num_shards];a.output.mkdir(parents=True,exist_ok=True);rows=[];sources=[]
 for e in entries:
  n=e['num_frames'] if not a.max_source_frames else min(e['num_frames'],a.max_source_frames)
  ids=np.arange(2 if a.grid==25 else 1,n,2 if a.grid==25 else 1,dtype=np.int64)
  paths=dict(tokenworld=a.artifacts/'tokenworld'/key(e)/'rgb.npz',ctrlworld=a.artifacts/f'ctrlworld_{a.grid}hz'/key(e)/'rgb.npz')
  candidates={}
  for name,path in paths.items():
   with np.load(path) as data:
    index={int(f):i for i,f in enumerate(data['frame_ids'])};assert set(ids)<=index.keys(),(name,key(e),ids)
    candidates[name]=data['rgb'][[index[int(f)] for f in ids]]
   meta=json.loads((path.parent/'metadata.json').read_text());sources.append(dict(model=name,dataset=e['dataset'],episode_index=e['episode_index'],rgb=str(path),rgb_sha256=sha(path),metadata=meta))
  reference=ground_truth(e,ids)
  for j,f in enumerate(ids):
   row=dict(dataset=e['dataset'],episode_index=e['episode_index'],task=e['task'],outcome=e['outcome'],panel_index=e['panel_index'],source_frame=int(f),seconds=float(f)/50,metrics={name:metric.frame(reference[j],rgb[j]) for name,rgb in candidates.items()})
   rows.append(row)
  # A small aligned sheet per episode; no temporally shifted or hand-selected frames.
  from PIL import Image,ImageDraw
  chosen=sorted(set([0,len(ids)//2,len(ids)-1]));sheet=Image.new('RGB',(3*228,len(chosen)*250),'white');draw=ImageDraw.Draw(sheet)
  for r,j in enumerate(chosen):
   for c,name in enumerate(['GT','Token-World','Ctrl-World']):
    image=reference[j] if c==0 else candidates['tokenworld' if c==1 else 'ctrlworld'][j]
    sheet.paste(Image.fromarray(image),(c*228,r*250+25));draw.text((c*228+3,r*250+4),f'{name} t={ids[j]/50:.2f}s',fill='black')
  sheet.save(a.output/f'{key(e)}.png');print('scored',key(e),len(ids),flush=True)
 atomic_json(dict(grid_hz=a.grid,rank=a.shard_index,world_size=a.num_shards,panel_sha256=sha(a.panel),pixel_metrics=metric.protocol(),rows=rows,sources=sources),a.output/f'scores.rank{a.shard_index:02d}.json')
if __name__=='__main__':main()
