#!/usr/bin/env python3
"""Validate paired coverage and summarize per-frame RGB scores."""
import argparse,json,sys,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_rgb.common import sha,atomic_json


def means(rows,model):
 fields=['psnr','ssim','lpips','mse_01'];values={f:float(np.mean([r['metrics'][model][f] for r in rows])) for f in fields}
 values['psnr_from_global_mse']=float(-10*np.log10(max(values['mse_01'],1e-12)));return values


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episode-indices');p.add_argument('--max-source-frames',type=int,default=0);p.add_argument('--grid',type=int,choices=[25,50],default=25);a=p.parse_args()
 panel=json.loads(a.panel.read_text());entries=panel['episodes']
 if a.episode_indices:
  wanted={int(x) for x in a.episode_indices.split(',')};entries=[e for e in entries if e['panel_index'] in wanted]
 files=sorted(a.scores.glob('scores.rank*.json'));assert files,'No metric shards';parts=[json.loads(p.read_text()) for p in files];assert len(parts)==parts[0]['world_size'];assert sorted(p['rank'] for p in parts)==list(range(len(parts)))
 for part in parts:assert part['grid_hz']==a.grid and part['pixel_metrics']==parts[0]['pixel_metrics'] and part['panel_sha256']==sha(a.panel)
 rows=sorted([r for p in parts for r in p['rows']],key=lambda r:(r['panel_index'],r['source_frame']));identity=lambda r:(r['dataset'],r['episode_index'],r['source_frame'])
 actual=[identity(r) for r in rows];assert len(set(actual))==len(actual)
 expected=set()
 for e in entries:
  n=e['num_frames'] if not a.max_source_frames else min(e['num_frames'],a.max_source_frames)
  expected.update((e['dataset'],e['episode_index'],f) for f in range(2 if a.grid==25 else 1,n,2 if a.grid==25 else 1))
 assert set(actual)==expected,dict(missing=len(expected-set(actual)),extra=len(set(actual)-expected))
 models=['tokenworld','ctrlworld'];episode_rows=[]
 for e in entries:
  rr=[r for r in rows if r['panel_index']==e['panel_index']]
  episode_rows.append({**{k:e[k] for k in ['panel_index','dataset','episode_index','task','outcome']},'scored_frames':len(rr),'metrics':{m:means(rr,m) for m in models}})
 aggregate={m:means(rows,m) for m in models};macro={m:{f:float(np.mean([e['metrics'][m][f] for e in episode_rows])) for f in ['psnr','ssim','lpips','mse_01']} for m in models}
 sources=[s for p in parts for s in p['sources']];current={}
 for m in models:
  contracts=[s['metadata']['contract'] for s in sources if s['model']==m]
  assert all(c==contracts[0] for c in contracts),f'{m} contracts differ'
  current[m]=contracts[0]
 full=len(entries)==len(panel['episodes']) and not a.max_source_frames
 selection_name=panel.get('selection','balanced100')
 scope=('full validation all500' if selection_name=='all500' else 'Table I balanced100 fixed subset') if full else f'subset diagnostic of {selection_name}'
 report=dict(version=1,scope=scope,selection=dict(panel_selection=selection_name,episodes=len(entries),tasks=len(set(e['task'] for e in entries)),outcomes=sorted(set(e['outcome'] for e in entries)),paired_frames=len(rows),common_hz=a.grid,tokenworld_generation_hz=50,source_frame0_excluded=True,temporal_interpolation=False),micro=aggregate,episode_macro=macro,by_outcome={o:{m:means([r for r in rows if r['outcome']==o],m) for m in models} for o in ['success','failure'] if any(r['outcome']==o for r in rows)},episodes=episode_rows,frames=rows,checkpoints_and_generation=current,metric_protocol=parts[0]['pixel_metrics'],provenance=dict(panel=str(a.panel),panel_sha256=sha(a.panel),metric_shards=[{'file':str(p),'sha256':sha(p)} for p in files],sources=sources,script_sha256={p.name:sha(p) for p in (ROOT/'scripts/compare_rgb').glob('*') if p.is_file()},historical_RGB_reused=any('source_metadata' in s['metadata'] for s in sources)),paper_checkpoint_identity='Current single-view Ctrl checkpoint is specified and hashed; identity with the weights behind the old paper row has not been established.')
 a.output.mkdir(parents=True,exist_ok=True);atomic_json(report,a.output/'report.json')
 with (a.output/'frames.csv').open('w') as f:
  fields=['dataset','episode_index','task','outcome','source_frame','seconds']+[m+'_'+v for m in models for v in ['psnr','ssim','lpips','mse_01']];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({**{k:r[k] for k in fields[:6]},**{m+'_'+v:r['metrics'][m][v] for m in models for v in ['psnr','ssim','lpips','mse_01']}})
 text=['# Paired RoboTwin RGB comparison','',report['scope'],'',f"Episodes: {len(entries)}; paired frames: {len(rows)}; common grid: {a.grid}Hz; no temporal interpolation.",'','|Model|Mean frame PSNR ↑|SSIM ↑|Alex-LPIPS v0.1 ↓|','|---|---:|---:|---:|']
 for m in models:text.append(f"|{m}|{aggregate[m]['psnr']:.4f}|{aggregate[m]['ssim']:.5f}|{aggregate[m]['lpips']:.5f}|")
 text+=['','Frame0 is conditioning only. Native model time steps, decoder checkpoint and numerical protocol are recorded in report.json.','The current Ctrl checkpoint has not been equated to the checkpoint behind the historical paper row.']
 (a.output/'summary.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(selection=report['selection'],micro=aggregate)),flush=True)
if __name__=='__main__':main()
