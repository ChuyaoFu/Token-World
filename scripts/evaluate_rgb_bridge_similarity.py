#!/usr/bin/env python3
"""Paired RGB bridge / adapter action agreement on the saved validation manifest."""
import argparse,json,hashlib,time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from tokenworld.codec.common import load_codec_checkpoint
from tokenworld.codec.deepstack_adapter import native_actions
from tokenworld.decoder.checkpoint import load_external_decoder
from tokenworld.data.dataset import RobotwinSVAETokenDataset
from tokenworld.evaluation import closed_loop as cl

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()

def measure(x,y,stats):
 xp=cl.policy_to_cache_order(x,stats);yp=cl.policy_to_cache_order(y,stats)
 x=x.double();y=y.double();j=list(cl.CONTINUOUS_INDICES);g=list(cl.GRIPPER_INDICES)
 return dict(normalized_action_cosine=F.cosine_similarity(x,y,dim=-1).mean().item(),normalized_joint_cosine=F.cosine_similarity(x[:,:12],y[:,:12],dim=-1).mean().item(),normalized_sse=(x-y).square().sum().item(),reference_energy=y.square().sum().item(),teacher_joint_mae_rad=(xp[:,j]-yp[:,j]).abs().mean().item(),teacher_gripper_agreement=(xp[:,g]==yp[:,g]).float().mean().item())

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--decoder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=2)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);dev=torch.device(a.device);torch.cuda.set_device(dev)
 config=json.loads((a.run_dir/'run.json').read_text())['contract']
 entries=[e for e in json.loads((a.run_dir/'manifest.json').read_text())['episodes'] if e['split']=='val'][a.rank::a.world_size]
 refs={}
 for path in sorted((a.run_dir/'action_similarity').glob('action_outputs.rank*.pt')):
  for r in torch.load(path,map_location='cpu',weights_only=False):
   key=(r['dataset'],r['episode_index'],r['frame']);assert key not in refs;refs[key]=r
 assert len(refs)==2000
 policy,_,_=cl.load_policy(Path(config['policy_bundle']),dev)
 codec,_=load_codec_checkpoint(Path(config['codec_checkpoint']),device=dev);codec.eval().requires_grad_(False)
 decoder=load_external_decoder(str(a.decoder),device=str(dev),freeze=True)
 ds=RobotwinSVAETokenDataset(config['cache'],raw_data_path=config['raw_data_root'],split='val',num_hist=1,num_pred=1,views=[0])
 lookup={(r['dataset'],r['episode_index']):i for i,r in enumerate(ds.records)}
 stats=json.loads((Path(config['policy_bundle'])/'model/dataset_statistics.json').read_text())
 rows=[];outputs=[];parity=[]
 for n,e in enumerate(entries):
  ri=lookup[(e['dataset'],e['episode_index'])];payload=ds._load_episode(ri)
  raw_images=ds._load_raw_images(ri,ds.records[ri],e['frames'])[:,0]
  for frame,raw in zip(e['frames'],raw_images):
   ref=refs[(e['dataset'],e['episode_index'],frame)];assert ref['task']==e['task'];teacher=ref['native_rgb_actions']
   z=payload['latents']['symmetric_qwen'][frame,:1].to(dev).float()
   merger=codec.decode(z,[108])[0].reshape(1,108,2560)
   with torch.autocast('cuda',dtype=torch.bfloat16):rgb=decoder(merger)[0]
   pixels=((rgb[0].float().cpu()+1)/2).clamp(0,1).permute(1,2,0).mul(255).round().to(torch.uint8).numpy()
   pil=Image.fromarray(pixels).resize((384,288),Image.Resampling.BILINEAR)
   result=policy.predict_action(examples=[{'image':[pil],'lang':e['task']+'\nTrajectory outcome: success.'}])
   q=torch.as_tensor(result['normalized_actions'],dtype=torch.float32).cpu().reshape(50,14)
   assert torch.isfinite(q).all()
   inputs=policy.qwen_vl_interface.build_qwenvl_inputs(images=[[pil]],instructions=[cl.action_instruction(policy,e['task'],'success')])
   inputs={k:v.to(dev) for k,v in inputs.items() if torch.is_tensor(v)}
   helper=native_actions(policy,inputs).cpu().reshape(50,14)
   rawpil=Image.fromarray(raw.numpy()).convert('RGB').resize((384,288),Image.Resampling.BILINEAR)
   original=policy.predict_action(examples=[{'image':[rawpil],'lang':e['task']+'\nTrajectory outcome: success.'}])
   replay=torch.as_tensor(original['normalized_actions'],dtype=torch.float32).cpu().reshape(50,14)
   if not rows:
    rawinputs=policy.qwen_vl_interface.build_qwenvl_inputs(images=[[rawpil]],instructions=[cl.action_instruction(policy,e['task'],'success')])
    rawinputs={k:v.to(dev) for k,v in rawinputs.items() if torch.is_tensor(v)}
    saved_replay=native_actions(policy,rawinputs).cpu().reshape(50,14)
    captured={}
    def capture_norm(module,args,output):
     captured['pre_norm']=args[0].detach();captured['post_norm']=output.detach()
    handle=policy.qwen_vl_interface.model.model.language_model.norm.register_forward_hook(capture_norm)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     native_out=policy.qwen_vl_interface(**rawinputs,output_attentions=False,output_hidden_states=True,return_dict=True)
    handle.remove()
    last=native_out.hidden_states[-1]
    parity.append(dict(bridge_helper_max_abs=(helper-q).abs().max().item(),native_api_vs_saved_teacher_max_abs=(replay-teacher).abs().max().item(),saved_teacher_replay_max_abs=(saved_replay-teacher).abs().max().item(),native_hidden_vs_pre_norm_max_abs=(last-captured['pre_norm']).abs().max().item(),native_hidden_vs_post_norm_max_abs=(last-captured['post_norm']).abs().max().item()))
    print(json.dumps({'parity':parity[-1]}),flush=True)
    assert parity[-1]['saved_teacher_replay_max_abs']<1e-5
    del captured,native_out,last
   row={k:e[k] for k in ['task','outcome','dataset','episode_index']};row.update(frame=frame,metrics={})
   for label,action in [('rgb_bridge',q),('adapter',ref['adapter_actions']),('codec_merger',ref['merger_only_actions']),('rgb_bridge_same_helper',helper)]:row['metrics'][label]=measure(action,teacher,stats)
   row['metrics']['rgb_bridge_vs_predict_action_teacher']=measure(q,replay,stats)
   row['metrics']['adapter_vs_predict_action_teacher']=measure(ref['adapter_actions'],replay,stats)
   row['metrics']['saved_teacher_vs_predict_action_teacher']=measure(teacher,replay,stats)
   rows.append(row);outputs.append(dict(dataset=e['dataset'],episode_index=e['episode_index'],frame=frame,rgb_bridge_actions=q,rgb_bridge_helper_actions=helper,predict_action_teacher=replay))
  if n%10==0 or n+1==len(entries):
   print(json.dumps(dict(episodes=n+1,total=len(entries),queries=len(rows))),flush=True)
 result=dict(status='complete',rows=rows,parity=parity,rank=a.rank,world_size=a.world_size,decoder=str(a.decoder),decoder_sha256=sha(a.decoder),reference='Saved RGB helper actions used for prior adapter similarity; helper uses final-normalized last_hidden_state. Extra metrics use actual predict_action teacher.',input='GT cached cam_high S-VAE latent; decoded RGB at every frame including frame 0; no WM predictions',metric='Mean cosine over 50 normalized 14D action vectors per query')
 (a.output/f'similarity.rank{a.rank:02d}.json').write_text(json.dumps(result,indent=2));torch.save(outputs,a.output/f'action_outputs.rank{a.rank:02d}.pt');print('complete',flush=True)
if __name__=='__main__':main()
