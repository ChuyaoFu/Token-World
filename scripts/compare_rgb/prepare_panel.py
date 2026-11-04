#!/usr/bin/env python3
"""Validate the Table I subset or all500 validation episodes for both models."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/table1_robotwin')]
import evaluate as table


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);base=ROOT.parent
 p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',choices=['balanced100','all500'],default='balanced100');p.add_argument('--panel',type=Path,default=ROOT/'scripts/table1_robotwin/panel.json');p.add_argument('--cache',type=Path,default=base/'IRASim/cache/robotwin_all50_symmetric_qwen_svae_dim16_50hz');p.add_argument('--ctrl-dataset',type=Path,default=base/'VA-CLIP/experiments/ctrlworld_robotwin_success_fail_cam_high/dataset/robotwin_success_fail_cam_high');a=p.parse_args()
 panel=json.loads(a.panel.read_text());index=json.loads((a.cache/'index.json').read_text());spec=json.loads((a.cache/'cache_spec.json').read_text())
 if a.selection=='balanced100':chosen=table.selected_reference_records(panel,index['records'],0)
 else:chosen=[(i,r) for i,r in enumerate(index['records']) if r['split']=='val']
 entries=[];tasks={}
 for i,(ri,r) in enumerate(chosen):
  payload=table._torch_load(a.cache/r['path']);raw=payload['actions'].float();states=payload['states'].float();ids=torch.as_tensor(payload['frame_indices']).long()
  assert torch.equal(raw,states);n=min(r['num_frames'],len(raw),len(ids));assert ids[:n].tolist()==list(range(n)),r
  cid=r['episode_index']+(2500 if r['dataset']=='robotwin_fail' else 0)
  found=[]
  for split in ['train','val']:
   ann=a.ctrl_dataset/'annotation'/split/f'{cid}.json';z=a.ctrl_dataset/'latent_videos'/split/str(cid)/'0.pt'
   if ann.is_file() and z.is_file():found.append((ann,z,split))
  assert len(found)==1,(r,found)
  ann,z,split=found[0];annotation=json.loads(ann.read_text())
  assert annotation['source_dataset']==r['dataset'] and annotation['source_episode_id']==r['episode_index']
  down=torch.as_tensor(annotation['states'],dtype=torch.float32)
  assert down.shape==raw[:n:2].shape and torch.equal(down,raw[:n:2]),(r,down.shape,raw.shape)
  gt_z=table._torch_load(z);assert len(gt_z)==len(down) and gt_z.shape[1:]==(4,32,40),(r,gt_z.shape)
  # Validate source videos are the same physical observations; differing paths may be aliases.
  rawvideo=Path(spec['data_root'])/r['dataset']/f'videos/chunk-{r["episode_index"]//1000:03d}/observation.images.cam_high/episode_{r["episode_index"]:06d}.mp4'
  ctrlvideo=a.ctrl_dataset/'videos'/split/str(cid)/'0.mp4'
  assert rawvideo.exists() and ctrlvideo.exists(),(rawvideo,ctrlvideo)
  entry=dict(panel_index=i,record_index=ri,dataset=r['dataset'],episode_index=r['episode_index'],task=r['task'],outcome=r['outcome'],cache_path=r['path'],num_frames=n,ctrl_episode_id=cid,ctrl_annotation=str(ann),ctrl_latent=str(z),ctrl_prepared_split=split,ctrl_text=annotation['texts'][0],source_video=str(rawvideo),ctrl_source_video=str(ctrlvideo),source_video_sha256=sha(rawvideo),ctrl_video_sha256=sha(ctrlvideo))
  assert entry['source_video_sha256']==entry['ctrl_video_sha256'],entry
  entries.append(entry);tasks.setdefault(r['task'],[]).append(r['outcome'])
 expected_per_outcome=1 if a.selection=='balanced100' else 5
 assert len(entries)==100*expected_per_outcome and len(tasks)==50
 assert all(v.count('success')==expected_per_outcome and v.count('failure')==expected_per_outcome for v in tasks.values())
 dense=sum(e['num_frames']-1 for e in entries);native=sum((e['num_frames']-1)//2 for e in entries)
 if a.selection=='balanced100':assert dense==22152,dense
 result=dict(version=1,panel=str(a.panel),panel_sha256=sha(a.panel),cache=str(a.cache),cache_spec=spec,ctrl_dataset=str(a.ctrl_dataset),episodes=entries,counts=dict(episodes=len(entries),tasks=50,predicted_50hz_frames=dense,shared_25hz_frames=native),alignment='source frame0 shared GT; score frame1..T-1 at50Hz or frame2,4,... at25Hz; no interpolation; identical source-video SHA256 and exact action/state match verified')
 if a.selection=='all500':result.update(selection='all500',panel=str(a.cache/'index.json'),panel_sha256=sha(a.cache/'index.json'))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result['counts']),flush=True)
if __name__=='__main__':main()
