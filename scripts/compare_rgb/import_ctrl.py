#!/usr/bin/env python3
"""Reuse native25Hz Ctrl-World replay RGB after matching immutable episode IDs."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_rgb.common import key,sha,atomic_json,save_npz,pixels


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=ROOT.parent/'TokenWorld/eval_outputs/robotwin_validation_replay_all500/ctrlworld_replay/ctrlworld');p.add_argument('--checkpoint',type=Path,default=ROOT.parent/'VA-CLIP/experiments/ctrlworld_robotwin_success_fail_cam_high/checkpoints/checkpoint-100000.pt');p.add_argument('--episode-indices');p.add_argument('--max-source-frames',type=int,default=0);a=p.parse_args();torch.set_num_threads(4)
 panel=json.loads(a.panel.read_text());found={}
 for meta in a.source.glob('*/metadata.json'):
  d=json.loads(meta.read_text());e=d['manifest_entry'];k=(e['dataset'],e['episode_index']);assert k not in found,k;found[k]=(meta,d)
 selected=panel['episodes']
 if a.episode_indices:
  wanted={int(x) for x in a.episode_indices.split(',')};selected=[e for e in selected if e['panel_index'] in wanted]
 checkpoint_sha=sha(a.checkpoint)
 audit=[]
 for e in selected:
  started=time.monotonic();source,m=found[(e['dataset'],e['episode_index'])]
  assert Path(m['checkpoint']).resolve()==a.checkpoint.resolve() and m['view']=='cam_high' and m['num_views']==1 and m['frame_rate_hz']==25 and m['sampling_steps']==50
  assert m['historical_ctrlworld_overlap_policy']=='keep_4_nonfinal_5_final'
  assert m['manifest_entry']['task']==e['task'] and m['manifest_entry']['outcome']==e['outcome']
  assert Path(m['source_annotation']).resolve()==Path(e['ctrl_annotation']).resolve()
  rgbpath=source.parent/'predicted_rgb.npy';rgb=np.load(rgbpath,mmap_mode='r');assert rgb.dtype==np.uint8 and rgb.shape[1:]==(256,320,3)
  assert len(rgb)==(e['num_frames']+1)//2,(e,len(rgb))
  n=e['num_frames'] if not a.max_source_frames else min(e['num_frames'],a.max_source_frames)
  ids=np.arange(2,n,2,dtype=np.int64);raw=np.asarray(rgb[ids//2]);out=[]
  for part in np.array_split(raw,max(1,(len(raw)+63)//64)):
   t=torch.from_numpy(part.copy()).permute(0,3,1,2).float()/127.5-1;out.append(pixels(t))
  d=a.output/'ctrlworld_25hz'/key(e);d.mkdir(parents=True,exist_ok=True);save_npz(d/'rgb.npz',frame_ids=ids,rgb=np.concatenate(out))
  metadata=dict(model='Ctrl-World',entry=e,predicted_frames=len(ids),source_metadata=str(source),source_metadata_sha256=sha(source),source_rgb=str(rgbpath),source_rgb_sha256=sha(rgbpath),source_generation_metadata=m,contract=dict(checkpoint=str(a.checkpoint),current_checkpoint_sha256=checkpoint_sha,historical_checkpoint_hash_recorded=False,checkpoint_mtime=a.checkpoint.stat().st_mtime,replay_metadata_mtime=source.stat().st_mtime,time_grid='native25Hz, source frame2,4,...; no interpolation',overlap='historical keep4 non-final/5 final; generated boundary reconstruction retained; physical time index validated',initial_GT_visual_frames=[0],output_frame0='generated reconstruction, excluded from all metrics',native_pixels='historical native RGB uint8; common bilinear AA224x224/rounding applied'),seconds=time.monotonic()-started)
  assert a.checkpoint.stat().st_mtime<=source.stat().st_mtime,'checkpoint newer than replay; regenerate instead'
  atomic_json(metadata,d/'metadata.json');audit.append(metadata);print('imported',key(e),len(ids),flush=True)
 atomic_json(dict(episodes=len(audit),frames=sum(e['predicted_frames'] for e in audit),checkpoint_sha256=checkpoint_sha,warning='Historical metadata lacks a generation-time checkpoint hash; validate current checkpoint by numerical replay probe.'),a.output/'ctrl_import_audit.json')
if __name__=='__main__':main()
