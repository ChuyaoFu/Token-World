#!/usr/bin/env python3
"""Supplement decoder training with other simulator seeds; hold out seed100014.

Held-out asset: 048_stapler model4 (seed100014). Training staplers, if any,
come from different instances (seed100003 model0, seed100013 model5).
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts/table1_robotwin')]
import evaluate as ev
from tokenworld.evaluation import closed_loop as cl
from tokenworld.codec.common import load_codec_checkpoint
from diagnose import sha


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(a.device)
    policy,_,_=cl.load_policy(ROOT.parent/'IRASim/results/action_expert/tokenworld_qwenoft_success_cam_high_teacher',torch.device(a.device))
    codec,_=load_codec_checkpoint(ROOT/'outputs/svae_d16_3views/best.pt',device=torch.device(a.device));codec.eval().requires_grad_(False)
    train=[c for c in torch.load(a.run/'train/probes.pt',map_location='cpu',weights_only=False) if c['split']=='train']
    validation=torch.load(a.run/'pilot/probes.pt',map_location='cpu',weights_only=False)
    seeds=[100000,100002,100003,100004,100005,100006,100007,100009,100010,100011,100012,100013]
    assert 100014 not in seeds and 100024 not in seeds
    lineage=[]
    for seed in seeds+[100024]:
        split='train' if seed in seeds else 'sim_val'
        source=ROOT/f'outputs/robotwin_full_cache_20261101/paired_visual_v1/put_object_cabinet/{seed}/adapter/frames'
        paths=sorted(source.glob('*.png'));assert paths,source
        # Exclude very late timeout frames; eight uniformly-spaced recorded frames.
        paths=[x for x in paths if int(x.stem)<=300]
        paths=[paths[i] for i in sorted(set(np.linspace(0,len(paths)-1,8 if split=='train' else 4).round().astype(int))) ]
        reset=ROOT/f'outputs/robotwin_wm_pair_20261101T080111Z/put_object_cabinet/{seed}/reset.json'
        assets=json.loads(reset.read_text())['physics']['task_assets']
        assert not (split=='train' and assets['selected_modelname']=='048_stapler' and assets['selected_model_id']=='4')
        for path in paths:
            rgb=torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2,0,1).float()[None]/127.5-1
            rgb=F.interpolate(rgb,size=(288,384),mode='bilinear',align_corners=False,antialias=True)
            f=ev.rgb_qwen_merger_features(policy,ev.to_pil(rgb),1)[:,0]
            with torch.autocast('cuda',dtype=torch.bfloat16):z=codec.encode(list(f));rec=torch.stack(codec.decode(z,[108]))
            c=dict(split=split,dataset='sim_cabinet',episode_index=seed,seed=seed,task='put_object_cabinet',outcome='recorded_policy_replay',frame=int(path.stem),target=rgb[0],raw=f[0].cpu().bfloat16(),reencoded=rec[0].cpu().bfloat16(),cached=rec[0].cpu().bfloat16(),source=str(path),source_sha256=sha(path),assets=assets,feature_origin='fresh raw RGB -> posterior mean -> S-VAE decoder; cached alias for training compatibility')
            (train if split=='train' else validation).append(c);lineage.append({k:v for k,v in c.items() if not torch.is_tensor(v)})
        print('prepared',split,seed,assets,len(paths),flush=True)
    torch.save(train,a.run/'probes_sim_train.pt');torch.save(validation,a.run/'probes_eval_extended.pt')
    guard=[dict(split='val',dataset=c.get('dataset','sim_cabinet'),episode_index=c.get('episode_index',c.get('seed'))) for c in validation]
    torch.save(guard,a.run/'validation_extended_guard.pt')
    (a.run/'sim_lineage.json').write_text(json.dumps({'train_seeds':seeds,'heldout_seeds':[100014,100024],'heldout_stapler_asset':['048_stapler','4'],'train_frames':len(train),'evaluation_frames':len(validation),'rows':lineage},indent=2))
    print('COMPLETE',len(train),len(validation),flush=True)
if __name__=='__main__':main()
