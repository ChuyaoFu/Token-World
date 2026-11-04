#!/usr/bin/env python3
"""Current-checkpoint Ctrl replay; native25Hz or explicitly forced dense50Hz."""
import argparse,json,sys,time,os,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from compare_rgb.common import sha,key,atomic_json,save_npz,pixels


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);base=ROOT.parent/'VA-CLIP'
 p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=base/'experiments/ctrlworld_robotwin_success_fail_cam_high/checkpoints/checkpoint-100000.pt');p.add_argument('--grid',type=int,choices=[25,50],default=25);p.add_argument('--device',default='cuda:0');p.add_argument('--episode-indices');p.add_argument('--max-source-frames',type=int,default=0);p.add_argument('--shard-index',type=int,default=int(os.environ.get('RANK',0)));p.add_argument('--num-shards',type=int,default=int(os.environ.get('WORLD_SIZE',1)));a=p.parse_args()
 if 'LOCAL_RANK' in os.environ:a.device=f'cuda:{os.environ["LOCAL_RANK"]}'
 torch.set_num_threads(4);torch.cuda.set_device(a.device);assert 0<=a.shard_index<a.num_shards
 panel=json.loads(a.panel.read_text());entries=panel['episodes'];stride=2 if a.grid==25 else 1
 if a.episode_indices:
  wanted={int(i) for i in a.episode_indices.split(',')};entries=[e for e in entries if e['panel_index'] in wanted]
 entries=entries[a.shard_index::a.num_shards]
 ctrl=base/'Ctrl-World';sys.path[:0]=[str(ctrl),str(ctrl/'scripts')]
 from config_robotwin_combined import wm_robotwin_combined_args
 from models.ctrl_world import CrtlWorld
 from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
 config=wm_robotwin_combined_args(task_type='replay');config.ckpt_path=str(a.checkpoint);config.val_model_path=str(a.checkpoint);config.svd_model_path=str(base/'assets/stable-video-diffusion-img2vid');config.clip_model_path=str(base/'assets/clip-vit-base-patch32');config.num_views=1;config.height=256;config.width=320;config.num_frames=5;config.num_history=6;config.pred_step=5;config.num_inference_steps=50;config.frame_level_cond=True;config.text_cond=True;config.dtype=torch.bfloat16
 contract=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),panel_sha256=sha(a.panel),source_fps=50,replay_grid_hz=a.grid,trained_native_hz=25,forced_dense50=(a.grid==50),num_views=1,view='cam_high',sampling_steps=50,added_time_fps=config.fps,history_idx=config.history_idx,overlap='native historical keep4 non-final/5 final; output0 excluded',seed='3047 + full cache record index',GT_visual_queries=[0],max_source_frames=a.max_source_frames,pixel_protocol='VAE [-1,1] -> shared bilinear AA224x224 -> round nearest uint8')
 pending=[]
 for e in entries:
  d=a.output/f'ctrlworld_{a.grid}hz'/key(e)
  if (d/'metadata.json').exists() and (d/'rgb.npz').exists():
   old=json.loads((d/'metadata.json').read_text());assert old['contract']==contract,(d,'contract mismatch')
   with np.load(d/'rgb.npz') as data:assert len(data['frame_ids'])==old['predicted_frames']
   print('skip',key(e),flush=True)
  else:pending.append(e)
 if not pending:return
 print('loading current Ctrl checkpoint',flush=True)
 model=CrtlWorld(config);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(state,strict=True);del state
 model.to(a.device).to(config.dtype).eval().requires_grad_(False);model.pipeline.set_progress_bar_config(disable=True)
 stat=json.loads((base/'experiments/ctrlworld_robotwin_success_fail_cam_high/dataset/dataset_meta_info/robotwin_success_fail_cam_high/stat.json').read_text());lo=np.asarray(stat['state_01'],dtype=np.float32)[None];hi=np.asarray(stat['state_99'],dtype=np.float32)[None]
 for e in pending:
  began=time.monotonic();anno=json.loads(Path(e['ctrl_annotation']).read_text());n=e['num_frames'] if not a.max_source_frames else min(e['num_frames'],a.max_source_frames)
  source_ids=np.arange(0,n,stride,dtype=np.int64)
  # Annotation.action contains all raw50Hz destination-aligned recorded states.
  full=np.asarray(anno['action'],dtype=np.float32);assert np.array_equal(full,np.asarray(anno['observation.state'],dtype=np.float32));states=full[source_ids]
  if stride==2:assert np.array_equal(states,np.asarray(anno['states'],dtype=np.float32)[:len(source_ids)])
  first=torch.load(e['ctrl_latent'],map_location='cpu',weights_only=True)[0:1].float().to(device=a.device,dtype=config.dtype)
  seed=3047+e['record_index'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
  visual=[first]*24;poses=[states[:1]]*24;times=[0]*24;chunks=[];all_rgb=[];all_z=[];interactions=(len(states)-2)//4+1
  for c in range(interactions):
   start=c*4;end=min(start+5,len(states));dest=states[start:end]
   if len(dest)<5:dest=np.concatenate([dest,np.repeat(dest[-1:],5-len(dest),axis=0)])
   hist=np.concatenate([poses[i] for i in config.history_idx]);actions=np.clip(2*(np.concatenate([hist,dest])-lo)/(hi-lo+1e-8)-1,-1,1)
   action=torch.from_numpy(actions)[None].to(device=a.device,dtype=config.dtype);history=torch.cat([visual[i] for i in config.history_idx])[None]
   text=model.action_encoder(action,anno['texts'][0],model.tokenizer,model.text_encoder)
   _,z=CtrlWorldDiffusionPipeline.__call__(model.pipeline,image=visual[-1],text=text,width=320,height=256,num_frames=5,history=history,num_inference_steps=50,decode_chunk_size=config.decode_chunk_size,max_guidance_scale=config.guidance_scale,fps=config.fps,motion_bucket_id=config.motion_bucket_id,mask=None,output_type='latent',return_dict=False,frame_level_cond=True)
   decoded=model.pipeline.vae.decode(z[0]/model.pipeline.vae.config.scaling_factor,num_frames=5).sample
   keep=5 if c==interactions-1 else 4;all_rgb.append(pixels(decoded[:keep]));all_z.append(z[0,:keep].float().cpu().numpy())
   boundary=min(start+4,len(states)-1);chunks.append(dict(chunk=c,source_start=int(source_ids[start]),condition_frame_ids=[times[i] for i in config.history_idx],future_action_frames=[int(source_ids[min(start+j,len(states)-1)]) for j in range(5)]))
   visual.append(z[0,4:5].detach());poses.append(states[boundary:boundary+1]);times.append(int(source_ids[boundary]))
   if c%10==0:print('chunk',key(e),c+1,'/',interactions,flush=True)
  rgb=np.concatenate(all_rgb)[:len(source_ids)][1:];latents=np.concatenate(all_z)[:len(source_ids)][1:];ids=source_ids[1:];assert len(rgb)==len(ids)
  d=a.output/f'ctrlworld_{a.grid}hz'/key(e);d.mkdir(parents=True,exist_ok=True);save_npz(d/'rgb.npz',frame_ids=ids,rgb=rgb);save_npz(d/'latents.npz',frame_ids=ids,latents=latents)
  atomic_json(dict(model='Ctrl-World',entry=e,contract=contract,predicted_frames=len(ids),chunks=chunks,seconds=time.monotonic()-began),d/'metadata.json')
  print(json.dumps(dict(rank=a.shard_index,episode=key(e),frames=len(ids),seconds=time.monotonic()-began)),flush=True)
if __name__=='__main__':main()
