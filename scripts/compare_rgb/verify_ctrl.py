#!/usr/bin/env python3
"""Replay the first Ctrl chunk and compare native pixels to the archived run."""
import argparse,json,sys,os,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--indices',default='0,50');a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(a.device)
 base=ROOT.parent/'VA-CLIP';ctrl=base/'Ctrl-World';sys.path[:0]=[str(ctrl),str(ctrl/'scripts')]
 from config_robotwin_combined import wm_robotwin_combined_args
 from models.ctrl_world import CrtlWorld
 from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
 config=wm_robotwin_combined_args(task_type='replay');ckpt=base/'experiments/ctrlworld_robotwin_success_fail_cam_high/checkpoints/checkpoint-100000.pt'
 config.ckpt_path=str(ckpt);config.val_model_path=str(ckpt);config.svd_model_path=str(base/'assets/stable-video-diffusion-img2vid');config.clip_model_path=str(base/'assets/clip-vit-base-patch32');config.num_views=1;config.height=256;config.width=320;config.num_frames=5;config.num_history=6;config.pred_step=5;config.num_inference_steps=50;config.frame_level_cond=True;config.text_cond=True;config.dtype=torch.bfloat16
 model=CrtlWorld(config);state=torch.load(ckpt,map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(state,strict=True);del state;model.to(a.device).to(config.dtype).eval().requires_grad_(False);model.pipeline.set_progress_bar_config(disable=True)
 stat=json.loads((base/'experiments/ctrlworld_robotwin_success_fail_cam_high/dataset/dataset_meta_info/robotwin_success_fail_cam_high/stat.json').read_text());lo=np.asarray(stat['state_01'],dtype=np.float32)[None];hi=np.asarray(stat['state_99'],dtype=np.float32)[None]
 archive=ROOT.parent/'TokenWorld/eval_outputs/robotwin_validation_replay_all500/ctrlworld_replay/ctrlworld';lookup={}
 for path in archive.glob('*/metadata.json'):
  m=json.loads(path.read_text());e=m['manifest_entry'];lookup[(e['dataset'],e['episode_index'])]=(path,m)
 entries=json.loads(a.panel.read_text())['episodes'];rows=[]
 for index in map(int,a.indices.split(',')):
  entry=entries[index];path,meta=lookup[(entry['dataset'],entry['episode_index'])];anno=json.loads(Path(entry['ctrl_annotation']).read_text());states=np.asarray(anno['states'],dtype=np.float32)
  initial=torch.load(entry['ctrl_latent'],map_location='cpu',weights_only=True)[0:1].float().to(device=a.device,dtype=config.dtype)
  seed=3047+meta['manifest_entry']['cache_record_index'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
  history=torch.cat([initial]*6)[None];pose=np.concatenate([states[:1]]*6+[states[:5]]);act=np.clip(2*(pose-lo)/(hi-lo+1e-8)-1,-1,1);act=torch.from_numpy(act)[None].to(device=a.device,dtype=config.dtype)
  text=model.action_encoder(act,anno['texts'][0],model.tokenizer,model.text_encoder);start=time.monotonic()
  _,z=CtrlWorldDiffusionPipeline.__call__(model.pipeline,image=initial,text=text,width=320,height=256,num_frames=5,history=history,num_inference_steps=50,decode_chunk_size=config.decode_chunk_size,max_guidance_scale=config.guidance_scale,fps=config.fps,motion_bucket_id=config.motion_bucket_id,mask=None,output_type='latent',return_dict=False,frame_level_cond=True)
  # Same temporal VAE decode and truncating uint8 conversion as the archive.
  rgb=model.pipeline.vae.decode(z[0]/model.pipeline.vae.config.scaling_factor,num_frames=5).sample
  rgb=((rgb/2+.5).clamp(0,1)*255).float().cpu().numpy().transpose(0,2,3,1).astype(np.uint8)
  old=np.load(path.parent/'predicted_rgb.npy',mmap_mode='r')[:4];diff=rgb[:4].astype(np.int16)-old.astype(np.int16)
  row=dict(panel_index=index,dataset=entry['dataset'],episode_index=entry['episode_index'],seed=seed,frames_compared=[0,2,4,6],exact=np.array_equal(rgb[:4],old),max_abs_uint8=int(np.abs(diff).max()),mean_abs_uint8=float(np.abs(diff).mean()),seconds=time.monotonic()-start,source_metadata=str(path))
  oldz=np.load(path.parent/'predicted_ctrl_latents.npy',mmap_mode='r')[:4];row['latent_max_abs']=float(np.abs(z[0,:4].float().cpu().numpy()-oldz).max());rows.append(row);print(json.dumps(row),flush=True)
 result=dict(checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),config=dict(history_idx=config.history_idx,sampling_steps=50,native_source_hz=25,added_time_fps=config.fps,num_views=1),rows=rows,all_exact=all(r['exact'] for r in rows),note='Archive metadata did not store a checkpoint hash. Exact replay, if observed, verifies current weights and runtime reproduce these sampled archived chunks.')
 a.output.write_text(json.dumps(result,indent=2))
 if not result['all_exact']:raise RuntimeError('Archived replay does not exactly match current checkpoint/runtime; inspect differences before reuse')
if __name__=='__main__':main()
