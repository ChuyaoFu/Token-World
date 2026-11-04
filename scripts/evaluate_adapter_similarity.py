#!/usr/bin/env python3
"""Recompute adapter action cosine against the full native RGB policy without persisting visual features."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tokenworld.codec.common import load_codec_checkpoint
from tokenworld.codec.deepstack_adapter import DeepStackAdapter,inject_actions,native_actions
from tokenworld.data.dataset import RobotwinSVAETokenDataset
from tokenworld.evaluation import closed_loop as cl


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()


@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',default='cuda:0');p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);dev=torch.device(a.device);torch.cuda.set_device(dev)
 run=json.loads((a.run_dir/'run.json').read_text());config=run['contract']
 checkpoint=a.run_dir/'adapter_best.pt';complete=json.loads((a.run_dir/'training_complete.json').read_text())
 assert sha(checkpoint)==complete['best_sha256']
 entries=[e for e in json.loads((a.run_dir/'manifest.json').read_text())['episodes'] if e['split']=='val'][a.shard_index::a.num_shards]
 previous=json.loads((a.run_dir/'action_evaluation.json').read_text())
 lookup_previous={(r['dataset'],r['episode_index'],r['frame']):r for r in previous['rows']}
 policy,_,_=cl.load_policy(Path(config['policy_bundle']),dev)
 codec,_=load_codec_checkpoint(Path(config['codec_checkpoint']),device=dev);codec.eval().requires_grad_(False)
 adapter=DeepStackAdapter().to(dev);adapter.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);adapter.eval().requires_grad_(False)
 ds=RobotwinSVAETokenDataset(config['cache'],raw_data_path=config['raw_data_root'],split='val',num_hist=1,num_pred=1,views=[0])
 lookup={(r['dataset'],r['episode_index']):i for i,r in enumerate(ds.records)}
 stats=json.loads((Path(config['policy_bundle'])/'model/dataset_statistics.json').read_text())
 rows=[];outputs=[];max_replay=0.
 for n,e in enumerate(entries):
  ri=lookup[(e['dataset'],e['episode_index'])];payload=ds._load_episode(ri)
  images=ds._load_raw_images(ri,ds.records[ri],e['frames'])[:,0]
  template=cl.build_policy_template(policy,e['task'],'success')
  for frame,raw in zip(e['frames'],images):
   pil=Image.fromarray(raw.numpy()).convert('RGB').resize((384,288),Image.Resampling.BILINEAR)
   inputs=policy.qwen_vl_interface.build_qwenvl_inputs(images=[[pil]],instructions=[cl.action_instruction(policy,e['task'],'success')])
   inputs={k:v.to(dev) for k,v in inputs.items() if torch.is_tensor(v)}
   teacher=native_actions(policy,inputs).cpu()[0]
   z=payload['latents']['symmetric_qwen'][frame,:1].to(dev).float()
   merger=codec.decode(z,[108])[0].reshape(1,108,2560).to(torch.bfloat16).float()
   with torch.autocast('cuda',dtype=torch.bfloat16):deep=adapter.features(z.to(torch.bfloat16).float(),merger)[0]
   predicted=inject_actions(policy,template,merger,deep).cpu()[0]
   baseline=inject_actions(policy,template,merger).cpu()[0]
   t_phys=cl.policy_to_cache_order(teacher,stats)
   row={k:e[k] for k in ['task','outcome','dataset','episode_index']};row.update(frame=frame,metrics={})
   for label,action in [('adapter',predicted),('codec_merger',baseline)]:
    x=action.double();y=teacher.double();err=x-y
    assert torch.isfinite(err).all()
    phys=cl.policy_to_cache_order(action,stats);j=list(cl.CONTINUOUS_INDICES);g=list(cl.GRIPPER_INDICES)
    metrics={'normalized_action_cosine':F.cosine_similarity(x,y,dim=-1).mean().item(),
       'normalized_joint_cosine':F.cosine_similarity(x[:,:12],y[:,:12],dim=-1).mean().item(),
       'normalized_sse':err.square().sum().item(),'reference_energy':y.square().sum().item(),
       'teacher_joint_mae_rad':(phys[:,j]-t_phys[:,j]).abs().mean().item(),
       'teacher_gripper_agreement':(phys[:,g]==t_phys[:,g]).float().mean().item()}
    old=lookup_previous[(e['dataset'],e['episode_index'],frame)]['metrics'][label]
    max_replay=max(max_replay,abs(metrics['teacher_joint_mae_rad']-old['teacher_joint_mae_rad']))
    row['metrics'][label]=metrics
   rows.append(row)
   outputs.append({'dataset':e['dataset'],'episode_index':e['episode_index'],'frame':frame,'task':e['task'],
                   'native_rgb_actions':teacher,'adapter_actions':predicted,'merger_only_actions':baseline})
  if n%10==0 or n+1==len(entries):print(json.dumps({'episodes':n+1,'total':len(entries),'queries':len(rows),'mae_replay_max_abs':max_replay}),flush=True)
 result={'status':'complete','checkpoint_sha256':complete['best_sha256'],'shard_index':a.shard_index,'num_shards':a.num_shards,
         'rows':rows,'mae_replay_max_abs':max_replay,'reference':'Native full Qwen3-VL RGB policy, all DeepStack streams; success prompt',
         'input':'GT cached cam_high S-VAE latent; no WM-generated latent','metric':'Mean cosine over 14D normalized action vectors, all 50 rows per query; not a success rate'}
 (a.output/f'similarity.rank{a.shard_index:02d}.json').write_text(json.dumps(result,indent=2))
 torch.save(outputs,a.output/f'action_outputs.rank{a.shard_index:02d}.pt')
 print('complete',flush=True)


if __name__=='__main__':main()
