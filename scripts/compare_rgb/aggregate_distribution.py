#!/usr/bin/env python3
"""Pool cached features across the exact selected episodes; compute FID/FVD."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_rgb.common import sha,key,atomic_json
from compare_rgb.frechet import distance,check


def main():
 p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--episode-indices');p.add_argument('--paired-report',type=Path);a=p.parse_args()
 panel=json.loads(a.panel.read_text());entries=panel['episodes']
 if a.episode_indices:
  wanted={int(x) for x in a.episode_indices.split(',')};entries=[e for e in entries if e['panel_index'] in wanted]
 generation_contracts={};generation_files=[]
 groups=['real','tokenworld','ctrlworld'];samples={g:{m:[] for m in ['fid','fvd16','fvd32']} for g in groups};metadata=[];contract=None;coverage={m:[] for m in ['fid','fvd16','fvd32']}
 for e in entries:
  for model,folder in [('tokenworld','tokenworld'),('ctrlworld','ctrlworld_25hz')]:
   generation=a.features.parent/folder/key(e)/'metadata.json';gm=json.loads(generation.read_text())
   for field in ['dataset','episode_index','record_index','num_frames']:assert gm['entry'][field]==e[field]
   assert gm['contract']['panel_sha256']==sha(a.panel)
   if model not in generation_contracts:generation_contracts[model]=gm['contract']
   else:assert generation_contracts[model]==gm['contract']
   generation_files.append(dict(model=model,metadata=str(generation),sha256=sha(generation)))
  ids=np.arange(2,e['num_frames'],2,dtype=np.int64)
  for g in groups:
   path=a.features/g/f'{key(e)}.npz';meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['feature_file_sha256']
   if contract is None:contract=meta['contract']
   else:assert meta['contract']==contract
   for field in ['dataset','episode_index','task','outcome','record_index','num_frames']:assert meta['entry'][field]==e[field]
   with np.load(path) as d:
    assert np.array_equal(d['frame_ids'],ids)
    for m in ['fid','fvd16','fvd32']:
     expected=len(ids) if m=='fid' else len(ids)//int(m[3:]);assert len(d[m])==expected
     if m!='fid':assert np.array_equal(d['clip_source_starts'+m[3:]],ids[::int(m[3:])][:expected])
     samples[g][m].append(d[m])
   metadata.append(dict(group=g,file=str(path),sha256=meta['feature_file_sha256'],rgb_sha256=meta['rgb_sha256']))
  for m in coverage:coverage[m].append(dict(dataset=e['dataset'],episode_index=e['episode_index'],count=len(ids) if m=='fid' else len(ids)//int(m[3:])))
 results={};counts={}
 for m in ['fid','fvd16','fvd32']:
  values={g:np.concatenate(samples[g][m]) for g in groups};counts[m]=len(values['real']);ddof=1 if m=='fid' else 0
  for g in groups:assert len(values[g])==counts[m]
  results[m]={g:distance(values['real'],values[g],ddof=ddof) for g in ['tokenworld','ctrlworld']}
  print(m,{g:v['value'] for g,v in results[m].items()},flush=True)
 full=len(entries)==500 and panel.get('selection')=='all500'
 report=dict(scope='all500 validation' if full else 'subset pipeline diagnostic; sample-biased and not the formal result',episodes=len(entries),tasks=len(set(e['task'] for e in entries)),sample_counts=counts,results=results,extractor_contract=contract,sampling=dict(common_hz=25,clip_frames=[16,32],clip_stride=[16,32],context_frame0=False,temporal_interpolation=False,cross_episode_clips=False,tail_padding=False,FID='all common predicted frames',FVD='all nonoverlapping full clips; partial tails dropped identically in all three groups'),statistics=dict(FID_covariance='N-1; pytorch-fid0.3.0 convention',FVD_covariance='N; StyleGAN-V feature-statistics convention',aggregation='one Frechet distance between pooled real/generated distributions; never average per-frame or per-episode FID/FVD',dependence='clips from the same trajectory are correlated; clip count is not independent trajectory count'),coverage=coverage,provenance=dict(panel=str(a.panel),panel_sha256=sha(a.panel),features=metadata,generation_contracts=generation_contracts,generation_metadata=generation_files,code_sha256={p.name:sha(p) for p in (ROOT/'scripts/compare_rgb').glob('*.py')}),numerical_checks=check())
 a.output.mkdir(parents=True,exist_ok=True);atomic_json(report,a.output/'report.json')
 lines=['# RoboTwin distribution metrics','',report['scope'],'',f"Episodes: {len(entries)}; frames: {counts['fid']}; clips16: {counts['fvd16']}; clips32: {counts['fvd32']}.",'','|Model|FID2048 ↓|FVD16 ↓|FVD32 ↓|','|---|---:|---:|---:|']
 for g in ['tokenworld','ctrlworld']:lines.append('|'+g+'|'+'|'.join(f"{results[m][g]['value']:.5f}" for m in ['fid','fvd16','fvd32'])+'|')
 lines+=['','FID/FVD measure marginal distributions; retain paired PSNR/SSIM/LPIPS for trajectory alignment.','Do not compare scores computed with different detectors, clip lengths, sampling, resolutions or sample counts.']
 (a.output/'summary.md').write_text('\n'.join(lines)+'\n')
 paired=a.paired_report or (a.output.parent/'report/report.json')
 if paired.is_file():
  original=json.loads(paired.read_text());assert original['selection']['episodes']==len(entries) and original['selection']['paired_frames']==counts['fid']
  assert original['provenance']['panel_sha256']==sha(a.panel)
  original['distribution_metrics']=dict(report=str(a.output/'report.json'),report_sha256=sha(a.output/'report.json'),results=results,sample_counts=counts,scope=report['scope'])
  atomic_json(original,paired)
if __name__=='__main__':main()
