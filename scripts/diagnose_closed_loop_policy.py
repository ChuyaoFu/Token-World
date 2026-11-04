"""Audit policy inputs and action conversion at recorded closed-loop query frames.
GT-image branches after t=0 are teacher-forced diagnostics, not closed-loop scores.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tokenworld.evaluation import closed_loop as cl
from tokenworld.data.dataset import RobotwinSVAETokenDataset
from tokenworld.codec.common import load_codec_checkpoint

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--replay-run',type=Path,required=True);p.add_argument('--raw-data-root',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dev=torch.device(a.device);torch.cuda.set_device(dev)
 entries=json.loads((a.replay_run/'manifest.json').read_text())['episodes']
 metas=[json.loads((a.replay_run/'closed_loop/episodes'/cl.episode_name(e)/'metadata.json').read_text()) for e in entries]
 cache=Path(metas[0]['cache']).parents[2]
 ds=RobotwinSVAETokenDataset(cache,raw_data_path=a.raw_data_root,split='val',num_hist=1,num_pred=1,views=[0],latent_method='symmetric_qwen')
 lookup={(r['dataset'],r['episode_index']):i for i,r in enumerate(ds.records)}
 policy,_,_=cl.load_policy(Path(metas[0]['policy_bundle']),dev)
 codec,_=load_codec_checkpoint(Path(metas[0]['codec_checkpoint']),device=dev);codec.eval().requires_grad_(False)
 stats=json.loads((Path(metas[0]['policy_bundle'])/'model/dataset_statistics.json').read_text())
 amin=torch.tensor(next(iter(stats.values()))['action']['min']);amax=torch.tensor(next(iter(stats.values()))['action']['max'])
 def correct(v):
  v=v.cpu();q=v.clone();q[...,:12]=(v[...,:12]+1)*.5*(amax[:12]-amin[:12])+amin[:12]
  q[...,12:]=(v[...,12:]>.49).float()
  return q[...,list(cl.QWEN_TO_CACHE)]
 joints=list(cl.CONTINUOUS_INDICES);grip=list(cl.GRIPPER_INDICES)
 def metrics(q,gt):
  q=q.cpu();gt=gt.cpu();d=q[:,joints]-gt[:,joints]
  return {'joint_mae_rad':d.abs().mean().item(),'joint_rmse_rad':d.square().mean().sqrt().item(),'first_joint_mae_rad':d[0].abs().mean().item(),'gripper_agreement':((q[:,grip]>.5)==(gt[:,grip]>.49)).float().mean().item()}
 rows=[];tensors=[]
 for e,meta in zip(entries,metas):
  ri=lookup[(e['dataset'],e['episode_index'])];record=ds.records[ri];payload=ds._load_episode(ri)
  saved=a.replay_run/'closed_loop/episodes'/cl.episode_name(e)
  pred=torch.load(saved/'predicted_svae_latents.pt',map_location='cpu',weights_only=True)
  used=torch.from_numpy(np.load(saved/'actions_closed_loop.npy'))
  query_ids=[x['frame'] for x in meta['query_trace']]
  images=ds._load_raw_images(ri,record,query_ids)[:,0]
  template=cl.build_policy_template(policy,e['task'],'success')
  for frame,raw in zip(query_ids,images):
   pil=Image.fromarray(raw.numpy()).convert('RGB').resize((384,288),Image.Resampling.BILINEAR)
   iface=policy.qwen_vl_interface;backbone=iface.model.model
   inputs=iface.build_qwenvl_inputs(images=[[pil]],instructions=[cl.action_instruction(policy,e['task'],'success')])
   inputs={k:v.to(dev) for k,v in inputs.items() if torch.is_tensor(v)}
   assert torch.equal(inputs['input_ids'],template['input_ids'].to(dev))
   with torch.autocast('cuda',dtype=torch.bfloat16): merged,deep=backbone.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])
   raw_features=torch.stack(list(merged)).unsqueeze(0)
   def native():
    with torch.autocast('cuda',dtype=torch.bfloat16):hidden=backbone(**inputs,use_cache=False,return_dict=True).last_hidden_state
    q=policy._gather_action_token_embeddings(hidden,inputs['input_ids'],action_token_id=policy.action_token_id)
    with torch.autocast('cuda',dtype=torch.float32):return policy.action_model.predict_action(q).float()
   native_full=native()
   merger=cl.policy_actions(policy,template,raw_features)
   if frame==0:
    old=backbone.get_image_features
    try:
     backbone.get_image_features=lambda *args,**kwargs:(merged,None)
     native_no_deep=native()
    finally:backbone.get_image_features=old
    parity=(native_no_deep-merger).abs().max().item()
   else:parity=None
   def decode(z):return torch.stack(list(codec.decode(z.to(dev).float(),[108])),dim=0).unsqueeze(0)
   codec_features=decode(payload['latents']['symmetric_qwen'][frame,:1])
   pred_features=decode(pred[frame])
   outputs={'native_full_raw':native_full,'merger_raw':merger,'merger_codec_gt':cl.policy_actions(policy,template,codec_features),'merger_closed_prediction':cl.policy_actions(policy,template,pred_features)}
   valid=min(50,len(pred)-frame-1)
   gt=payload['actions'][frame:frame+valid].float()
   row={'task':e['task'],'source_outcome':e['outcome'],'episode_index':e['episode_index'],'frame':frame,'valid_actions':valid,'native_no_deep_vs_injection_max_abs':parity,'metrics':{}}
   tensorrow={k:row[k] for k in ['task','source_outcome','episode_index','frame']};tensorrow['gt_actions']=gt
   for label,v in outputs.items():
    v=v[0,:valid].cpu();legacy=cl.policy_to_cache_order(v,stats,gripper_mode="legacy_minmax").cpu();fixed=correct(v)
    row['metrics'][label]={'legacy_conversion':metrics(legacy,gt),'correct_conversion':metrics(fixed,gt),'gripper_flip_fraction':((legacy[:,grip]>.5)!=(fixed[:,grip]>.5)).float().mean().item()}
    tensorrow[label]=v
    if label=='merger_closed_prediction':row['replayed_actions_vs_saved_max_abs']=(legacy-used[frame:frame+valid]).abs().max().item()
   rows.append(row);tensors.append(tensorrow)
  print('audited',e['task'],e['outcome'],len(query_ids),'queries',flush=True)
 summary={}
 for subset,chosen in [('initial_all',[r for r in rows if r['frame']==0]),('initial_success',[r for r in rows if r['frame']==0 and r['source_outcome']=='success']),('all_queries',rows),('success_queries',[r for r in rows if r['source_outcome']=='success'])]:
  summary[subset]={label:{conv:{key:float(np.mean([r['metrics'][label][conv][key] for r in chosen])) for key in chosen[0]['metrics'][label][conv]} for conv in ['legacy_conversion','correct_conversion']} for label in chosen[0]['metrics']}
 weighted_flip=sum(r['metrics']['merger_closed_prediction']['gripper_flip_fraction']*r['valid_actions']*2 for r in rows)/sum(r['valid_actions']*2 for r in rows)
 result={'status':'complete','scope':'28 query states on six episodes; GT branches are teacher-forced after frame zero; recorded actions on failure episodes are not a success target','rows':rows,'summary':summary,'legacy_gripper_flip_fraction_weighted':weighted_flip,'gripper_normalization':'joint min_max [-1,1], gripper binary [0,1], threshold .49','replayed_actions_max_abs':max(r['replayed_actions_vs_saved_max_abs'] for r in rows)}
 (a.output/'policy_audit.json').write_text(json.dumps(result,indent=2));torch.save(tensors,a.output/'policy_actions.pt')
 print('SUMMARY',json.dumps(summary),flush=True);print('flip',weighted_flip,'replay',result['replayed_actions_max_abs'],flush=True)
if __name__=='__main__':main()
