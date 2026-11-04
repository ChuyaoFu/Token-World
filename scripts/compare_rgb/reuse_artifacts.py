#!/usr/bin/env python3
"""Rebind identical current-checkpoint episode artifacts to a larger panel.

Panel membership affects only selection, not model inputs or per-episode RNG.
The original generation metadata remains explicitly preserved in provenance.
"""
import argparse,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_rgb.common import sha,key,atomic_json


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--source-panel',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 old=json.loads(a.source_panel.read_text());new=json.loads(a.panel.read_text())
 assert old['cache']==new['cache'] and old['cache_spec']==new['cache_spec'] and old['ctrl_dataset']==new['ctrl_dataset']
 old_lookup={(e['dataset'],e['episode_index']):e for e in old['episodes']};old_hash=sha(a.source_panel);new_hash=sha(a.panel);reused=[]
 assert a.source_root.resolve()!=a.output.resolve(),'Use a distinct destination root'
 for e in new['episodes']:
  old_e=old_lookup.get((e['dataset'],e['episode_index']))
  if old_e is None:continue
  assert {k:v for k,v in e.items() if k!='panel_index'}=={k:v for k,v in old_e.items() if k!='panel_index'},e
  for model in ['tokenworld','ctrlworld_25hz']:
   source=a.source_root/model/key(e);mp=source/'metadata.json'
   if not mp.is_file():continue
   metadata=json.loads(mp.read_text())
   assert 'source_metadata' not in metadata,'Historical archive imports are not current replay artifacts'
   assert metadata['entry']==old_e and metadata['contract']['panel_sha256']==old_hash
   assert metadata['contract']['max_source_frames']==0
   destination=a.output/model/key(e);destination.mkdir(parents=True,exist_ok=True)
   changed=json.loads(json.dumps(metadata));changed['entry']=e;changed['contract']['panel_sha256']=new_hash
   changed['reuse_provenance']=dict(source_metadata=str(mp.resolve()),source_metadata_sha256=sha(mp),source_panel=str(a.source_panel),source_panel_sha256=old_hash,original_generation_contract=metadata['contract'],verified='All episode inputs and record_index/seed inputs identical; only selection panel and panel_index changed; RGB and latents are linked unchanged.')
   if (destination/'metadata.json').exists():
    existing=json.loads((destination/'metadata.json').read_text());assert existing['contract']==changed['contract'] and existing['entry']==e
   for name in ['rgb.npz','latents.npz']:
    src=(source/name).resolve();dst=destination/name;assert src.is_file(),src
    if dst.exists():assert sha(src)==sha(dst)
    else:
     try:os.link(src,dst)
     except OSError:dst.symlink_to(src)
   atomic_json(changed,destination/'metadata.json');reused.append(dict(model=model,dataset=e['dataset'],episode_index=e['episode_index'],old_panel_index=old_e['panel_index'],new_panel_index=e['panel_index']))
 atomic_json(dict(source_root=str(a.source_root),source_panel_sha256=old_hash,panel_sha256=new_hash,reused=reused),a.output/'reuse_audit.json')
 print('reused model/episode artifacts',len(reused),flush=True)
if __name__=='__main__':main()
