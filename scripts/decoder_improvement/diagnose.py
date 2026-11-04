#!/usr/bin/env python3
"""Build immutable, split-disjoint RGB/merger probes; no decoded full cache."""
import argparse, json, os, sys, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/table1_robotwin')]
import evaluate as ev
from tokenworld.evaluation import closed_loop as cl
from tokenworld.codec.common import load_codec_checkpoint


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def args():
    p=argparse.ArgumentParser(description=__doc__)
    base=ROOT.parent/'IRASim'
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=base/'cache/robotwin_all50_symmetric_qwen_svae_dim16_50hz')
    p.add_argument('--data-root',type=Path,default=base/'robotdata/robotwin-success-fail-lerobot')
    p.add_argument('--policy-bundle',type=Path,default=base/'results/action_expert/tokenworld_qwenoft_success_cam_high_teacher')
    p.add_argument('--codec',type=Path,default=ROOT/'outputs/svae_d16_3views/best.pt')
    p.add_argument('--decoder',type=Path,default=ROOT/'outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt')
    p.add_argument('--wm',type=Path,default=ROOT/'outputs/world_model_svae_d16_cam_high_100k/checkpoint_step_0100000.pt')
    p.add_argument('--train-episodes',type=int,default=24)
    p.add_argument('--val-episodes',type=int,default=8)
    p.add_argument('--frames',type=int,default=4)
    p.add_argument('--skip-wm',action='store_true')
    p.add_argument('--device',default='cuda:0')
    return p.parse_args()


def select_records(records,n,split):
    # One episode per task/outcome before repeating any task; stable task order.
    pool=[(i,r) for i,r in enumerate(records) if r['split']==split]
    preferred=['put_object_cabinet','click_alarmclock','adjust_bottle','beat_block_hammer','place_can_basket','stack_blocks_two']
    pool.sort(key=lambda ir:(preferred.index(ir[1]['task']) if ir[1]['task'] in preferred else len(preferred),ir[1]['task'],ir[1]['outcome'],ir[1]['episode_index']))
    chosen=[];seen=set()
    for i,r in pool:
        key=(r['task'],r['outcome'])
        if key in seen:continue
        chosen.append((i,r));seen.add(key)
        if len(chosen)==n:break
    return chosen


@torch.inference_mode()
def main():
    a=args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'probes.pt').exists():raise FileExistsError(a.output/'probes.pt')
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.set_device(a.device)
    index=json.loads((a.cache/'index.json').read_text());spec=json.loads((a.cache/'cache_spec.json').read_text())
    policy,_,_=cl.load_policy(a.policy_bundle,torch.device(a.device))
    codec,cp=load_codec_checkpoint(a.codec,device=torch.device(a.device));codec.eval().requires_grad_(False)
    rawds=ev.RawCacheDatasetAdapter(spec,a.data_root)
    probes=[];selected={s:select_records(index['records'],n,s) for s,n in [('train',a.train_episodes),('val',a.val_episodes)] if n}
    for split,records in selected.items():
        for ri,r in records:
            payload=ev._torch_load(a.cache/r['path']);n=min(r['num_frames'],len(payload['frame_indices']))
            ids=sorted(set(np.linspace(1,n-1,a.frames).round().astype(int).tolist()))
            rgb=torch.cat([ev.raw_cam_high_rgb(rawds,r,payload,i,i+1) for i in ids])
            features=ev.rgb_qwen_merger_features(policy,ev.to_pil(rgb),4)[:,0]
            assert features.shape[1:]==(108,2560)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=codec.encode(list(features));rec=torch.stack(codec.decode(z,[108]*len(z)))
                cached=torch.stack(codec.decode(payload['latents']['symmetric_qwen'][ids,0].float().to(a.device),[108]*len(ids)))
            for j,i in enumerate(ids):
                probes.append(dict(split=split,task=r['task'],outcome=r['outcome'],dataset=r['dataset'],episode_index=r['episode_index'],record_index=ri,frame=i,source_frame=int(payload['frame_indices'][i]),target=rgb[j].cpu(),raw=features[j].cpu().bfloat16(),reencoded=rec[j].cpu().bfloat16(),cached=cached[j].cpu().bfloat16()))
            print('prepared',split,r['task'],r['outcome'],r['episode_index'],ids,'cache_feature_mse',float((rec.float()-cached.float()).square().mean()),flush=True)
    # Keep the known simulator seed separate from both dataset splits.
    known=ROOT/'outputs/robotwin_full_cache_20261101/paired_visual_v1/put_object_cabinet/100014/adapter/frames'
    if known.is_dir():
        for step in [0,50,150,250]:
            path=known/f'{step:04d}.png'
            if not path.exists():continue
            image=Image.open(path).convert('RGB');array=np.asarray(image).copy()
            rgb=torch.from_numpy(array).permute(2,0,1).float()[None]/127.5-1
            rgb=F.interpolate(rgb,size=(288,384),mode='bilinear',align_corners=False,antialias=True)
            f=ev.rgb_qwen_merger_features(policy,ev.to_pil(rgb),1)[:,0]
            with torch.autocast('cuda',dtype=torch.bfloat16):z=codec.encode(list(f));rec=torch.stack(codec.decode(z,[108]))
            wm=ROOT/f'outputs/robotwin_gru_only_cabinet_v1/matched/100014/frames/{step:04d}.png'
            probe=dict(split='known_seed',task='put_object_cabinet',seed=100014,frame=step,source=str(path),source_sha256=sha(path),target=rgb[0],raw=f[0].cpu().bfloat16(),reencoded=rec[0].cpu().bfloat16(),roi_xyxy_224=[82,51,173,115])
            if wm.exists():
                im=np.asarray(Image.open(wm).convert('RGB')).copy();probe['historical_wm_rgb']=torch.from_numpy(im).permute(2,0,1).float()/127.5-1
                probe['historical_wm_source']=str(wm)
            probes.append(probe)
    del policy;torch.cuda.empty_cache()
    if not a.skip_wm and selected.get('val'):
        rollout=ev.WorldModelRollout(str(a.wm),device=a.device,num_steps=16,no_shortcut=False)
        stats=ev._torch_load(a.cache/'train_stats.pt')
        scope=rollout.config.get('data',{}).get('latent_stats_scope','global')
        mean,std,_,_=ev.resolve_latent_normalization(stats,latent_method='symmetric_qwen',video_keys=spec['video_keys'],view_indices=[0],scope=scope)
        mean=mean.to(a.device);std=std.to(a.device);am=stats['action']['mean'].float().to(a.device);ast=stats['action']['std'].float().to(a.device).clamp_min(1e-6)
        for ri,r in selected['val']:
            payload=ev._torch_load(a.cache/r['path']);actions=payload['actions'].float();states=payload['states'].float();assert torch.equal(actions,states)
            z=payload['latents']['symmetric_qwen'][0:1,:1].float().to(a.device);z=((z-mean)/std)[None]
            prop=((states[0:1].to(a.device)-am)/ast)[None];history=None;start=0;chunk=0
            wanted={p['frame']:p for p in probes if p['split']=='val' and p['record_index']==ri}
            while start<max(wanted):
                count=min(15,len(actions)-1-start);act=actions[start:start+count+1].to(a.device)
                cont=(act[:,list(ev.RobotwinSVAETokenDataset.CONTINUOUS_INDICES)]-am[list(ev.RobotwinSVAETokenDataset.CONTINUOUS_INDICES)])/ast[list(ev.RobotwinSVAETokenDataset.CONTINUOUS_INDICES)]
                disc=(act[:,list(ev.RobotwinSVAETokenDataset.GRIPPER_INDICES)]>.5).long();torch.manual_seed(3047+ri*1000+chunk)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result=rollout.rollout(z,discrete_actions=disc[None],continuous_actions=cont[None],init_proprio=prop,time_cache=history,return_time_cache=True,gru_only_boundary=True)
                    pred=result.latents[:,1:count+1].float()*std+mean
                    for frame in sorted(wanted):
                        if start<frame<=start+count:
                            tok=pred[0,frame-start-1].reshape(1,108,16)
                            wanted[frame]['wm']=torch.stack(codec.decode(tok,[108]))[0].cpu().bfloat16()
                history=result.time_cache
                assert history.next_kv_cache is None
                z=result.latents[:,count:count+1].detach();prop=result.proprio[:,count:count+1].detach();start+=count;chunk+=1
            print('wm prepared',r['task'],r['outcome'],len(wanted),flush=True)
    torch.save(probes,a.output/'probes.pt')
    manifest={'version':1,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'checkpoints':{k:{'path':str(v),'sha256':sha(v)} for k,v in [('decoder',a.decoder),('codec',a.codec),('wm',a.wm)]},'panel_sha256':sha(ROOT/'scripts/table1_robotwin/panel.json'),'cache_spec':spec,'codec_config':cp['codec_config'],'records':[{k:v for k,v in p.items() if not torch.is_tensor(v)} for p in probes],'input_contract':{'merger':[108,2560],'grid':[9,12],'order':'row-major post-spatial-merge Qwen merger','decoder_normalization':'learned per-token LayerNorm; no latent standardization on 2560D merger','wm_normalization':'checkpoint latent stats applied once before WM, inverted once before codec','target_resize':'288x384 -> bilinear align_corners=False 224x224; no crop','rgb_range':[-1,1],'wm':'release100k, intra-segment KV, cross-segment GRU only; recorded action destination alignment'}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('COMPLETE',len(probes),flush=True)
if __name__=='__main__':main()
