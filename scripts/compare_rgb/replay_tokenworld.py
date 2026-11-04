#!/usr/bin/env python3
"""Generate each 50Hz Token-World trajectory once, retaining RGB and latent."""
import argparse,json,sys,time,os
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_rgb.common import table,sha,key,atomic_json,save_npz,pixels
from tokenworld.codec.common import load_codec_checkpoint
from tokenworld.decoder.checkpoint import load_external_decoder


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/world_model_svae_d16_cam_high_100k/checkpoint_step_0100000.pt');p.add_argument('--codec',type=Path,default=ROOT/'outputs/svae_d16_3views/best.pt');p.add_argument('--decoder',type=Path,default=ROOT/'outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt');p.add_argument('--device',default='cuda:0');p.add_argument('--shard-index',type=int,default=int(os.environ.get('RANK',0)));p.add_argument('--num-shards',type=int,default=int(os.environ.get('WORLD_SIZE',1)));p.add_argument('--episode-indices');p.add_argument('--max-source-frames',type=int,default=0);a=p.parse_args()
 if 'LOCAL_RANK' in os.environ:a.device=f'cuda:{os.environ["LOCAL_RANK"]}'
 assert 0<=a.shard_index<a.num_shards
 torch.set_num_threads(4);torch.cuda.set_device(a.device);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 panel=json.loads(a.panel.read_text());cache=Path(panel['cache']);spec=panel['cache_spec'];entries=panel['episodes']
 if a.episode_indices:
  wanted={int(v) for v in a.episode_indices.split(',')};entries=[e for e in entries if e['panel_index'] in wanted]
 entries=entries[a.shard_index::a.num_shards]
 contract=dict(panel_sha256=sha(a.panel),wm=str(a.checkpoint),wm_sha256=sha(a.checkpoint),codec=str(a.codec),codec_sha256=sha(a.codec),decoder=str(a.decoder),decoder_sha256=sha(a.decoder),num_denoise_steps=16,chunk_destinations=15,cross_segment_cache='GRU only; KV retained within segment',source_fps=50,seed='3047 + full_cache_record_index*1000 + chunk',initial_GT_visual_frames=[0],action_alignment='boundary action plus destination actions; recorded action==state; recursive predicted latent/proprio',max_source_frames=a.max_source_frames,pixel_protocol='fixed [-1,1] -> rounded uint8, full-image bilinear AA224x224',latent_storage='float32, unnormalized, no additional quantization')
 pending=[]
 for e in entries:
  d=a.output/'tokenworld'/key(e);meta=d/'metadata.json'
  if meta.exists() and (d/'rgb.npz').exists() and (d/'latents.npz').exists():
   old=json.loads(meta.read_text());assert old['contract']==contract,(meta,'contract differs')
   with np.load(d/'rgb.npz') as data:assert len(data['frame_ids'])==old['predicted_frames']
   print('skip',key(e),flush=True)
  else:pending.append(e)
 if not pending:return
 rollout=table.WorldModelRollout(str(a.checkpoint),device=a.device,num_steps=16,no_shortcut=False)
 stats=table._torch_load(cache/'train_stats.pt');scope=rollout.config.get('data',{}).get('latent_stats_scope','global')
 mean,std,resolved,keys=table.resolve_latent_normalization(stats,latent_method='symmetric_qwen',video_keys=spec['video_keys'],view_indices=[0],scope=scope)
 mean=mean.to(a.device);std=std.to(a.device);am=stats['action']['mean'].float().to(a.device);ast=stats['action']['std'].float().to(a.device).clamp_min(1e-6)
 ci=list(table.RobotwinSVAETokenDataset.CONTINUOUS_INDICES);gi=list(table.RobotwinSVAETokenDataset.GRIPPER_INDICES)
 codec,_=load_codec_checkpoint(a.codec,device=torch.device(a.device));codec.eval().requires_grad_(False);decoder=load_external_decoder(str(a.decoder),a.device,freeze=True)
 for e in pending:
  start_time=time.monotonic();d=a.output/'tokenworld'/key(e);d.mkdir(parents=True,exist_ok=True)
  payload=table._torch_load(cache/e['cache_path']);actions=payload['actions'].float();states=payload['states'].float();assert torch.equal(actions,states)
  n=e['num_frames'] if not a.max_source_frames else min(e['num_frames'],a.max_source_frames)
  initial=payload['latents']['symmetric_qwen'][0:1,:1].float().to(a.device)
  z=((initial-mean)/std)[None];prop=((states[0:1].to(a.device)-am)/ast)[None];history=None;t=0;chunk=0;all_z=[];all_rgb=[];trace=[]
  while t<n-1:
   count=min(15,n-1-t);act=actions[t:t+count+1].to(a.device)
   continuous=(act[:,ci]-am[ci])/ast[ci];discrete=(act[:,gi]>.5).long();torch.manual_seed(3047+e['record_index']*1000+chunk)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    result=rollout.rollout(z,discrete_actions=discrete[None],continuous_actions=continuous[None],init_proprio=prop,time_cache=history,return_time_cache=True,gru_only_boundary=True)
    assert result.latents.shape[1]==count+1
    raw=result.latents[:,1:count+1].float()*std+mean
    features=table.decode_svae_tokens(codec,raw).reshape(count,108,2560)
    rgb=torch.cat([decoder(f)[0] for f in features.split(4)])
   all_rgb.append(pixels(rgb));all_z.append(raw[0,:,0].cpu().float().numpy())
   history=result.time_cache;assert history.next_kv_cache is None
   trace.append(dict(chunk=chunk,boundary=t,destinations=list(range(t+1,t+count+1)),carried_KV=False))
   z=result.latents[:,count:count+1].detach();prop=result.proprio[:,count:count+1].detach();t+=count;chunk+=1
  ids=np.arange(1,n,dtype=np.int64);values=np.concatenate(all_rgb);latents=np.concatenate(all_z);assert len(values)==len(ids)==n-1
  save_npz(d/'rgb.npz',frame_ids=ids,rgb=values);save_npz(d/'latents.npz',frame_ids=ids,latents=latents)
  meta=dict(model='Token-World',entry=e,contract=contract,predicted_frames=len(ids),normalization=dict(scope=resolved,keys=keys),chunks=trace,seconds=time.monotonic()-start_time)
  atomic_json(meta,d/'metadata.json');print(json.dumps(dict(rank=a.shard_index,episode=key(e),predicted_frames=len(ids),seconds=meta['seconds'])),flush=True)
if __name__=='__main__':main()
