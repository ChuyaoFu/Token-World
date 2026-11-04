"""Isolated closed-loop interventions; production evaluator stays unchanged."""
import argparse,json,sys
from pathlib import Path
import torch
from PIL import Image
from tokenworld.evaluation import closed_loop as cl
from tokenworld.data.dataset import RobotwinSVAETokenDataset
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--diagnostic-variant',choices=['gripper','rgb_gripper'],required=True)
p.add_argument('--raw-data-root',type=Path)
opt,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
original_convert=cl.policy_to_cache_order

def correct_convert(values,stats,**kwargs):
 result=original_convert(values,stats,gripper_mode="legacy_minmax")
 result[...,list(cl.GRIPPER_INDICES)]=(values[...,12:]>.49).to(result.dtype)
 return result
cl.policy_to_cache_order=correct_convert
original_run=cl.run_episode
state={}
@torch.inference_mode()
def rgb_actions(policy,template,features):
 if state['queries']==0:
  pil=state['initial_image']
 else:
  with torch.autocast('cuda',dtype=torch.bfloat16):
   rgb=state['decoder'](features.reshape(-1,108,2560))[0]
  pixels=((rgb[0].float().cpu()+1)/2).clamp(0,1).permute(1,2,0).mul(255).round().to(torch.uint8).numpy()
  pil=Image.fromarray(pixels).resize((384,288),Image.Resampling.BILINEAR)
 state['queries']+=1
 # Official policy path includes Qwen3-VL DeepStack; no future GT is used.
 result=policy.predict_action(examples=[{'image':[pil],'lang':state['task']+'\nTrajectory outcome: success.'}])
 return torch.as_tensor(result['normalized_actions'],device=next(policy.parameters()).device,dtype=torch.float32)

def run(entry,args,**kw):
 if args.gripper_mode!="binary":raise ValueError("Diagnostic interventions require --gripper-mode binary")
 if opt.diagnostic_variant=='rgb_gripper':
  if 'dataset' not in state:
   state['dataset']=RobotwinSVAETokenDataset(args.cache,raw_data_path=opt.raw_data_root,split='val',num_hist=1,num_pred=1,views=[0],latent_method='symmetric_qwen')
  ds=state['dataset'];ri=next(i for i,r in enumerate(ds.records) if r['dataset']==entry['dataset'] and r['episode_index']==entry['episode_index'])
  raw=ds._load_raw_images(ri,ds.records[ri],[0])[0,0].numpy()
  state.update(initial_image=Image.fromarray(raw).resize((384,288),Image.Resampling.BILINEAR),queries=0,task=entry['task'],decoder=kw['rgb_decoder'])
 result=original_run(entry,args,**kw)
 result['diagnostic_variant']=opt.diagnostic_variant
 result['gripper_conversion']='binary threshold 0.49; joint minmax unchanged'
 result['policy_visual_interface']='native RGB with DeepStack; raw GT frame 0, predicted decoded RGB afterwards' if opt.diagnostic_variant=='rgb_gripper' else 'original merger-only'
 (args.output_root/'episodes'/cl.episode_name(entry)/'metadata.json').write_text(json.dumps(result,indent=2))
 return result
cl.run_episode=run
if opt.diagnostic_variant=='rgb_gripper':
 if opt.raw_data_root is None:raise ValueError('--raw-data-root required for RGB diagnostic')
 cl.policy_actions=rgb_actions
cl.main()
