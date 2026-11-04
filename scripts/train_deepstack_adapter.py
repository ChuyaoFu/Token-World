#!/usr/bin/env python3
"""Small cam_high pilot: frozen S-VAE latent -> three policy DeepStack streams.
No World Model, codec, RGB decoder or policy weights are updated. Validation
uses held-out episodes, not closed-loop task success. Input latents are GT
posterior means from the existing cache. Training loss is feature-only.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from tokenworld.codec.common import load_codec_checkpoint, evenly_spaced_indices
from tokenworld.data.dataset import RobotwinSVAETokenDataset
from tokenworld.evaluation import closed_loop as cl

TASKS = ['put_object_cabinet', 'shake_bottle_horizontally', 'turn_switch']


def save_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)


def event(out, **row):
    row['time'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    print(json.dumps(row), flush=True)
    with (out / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    save_json(out / 'status.json', row)


class DeepStackAdapter(nn.Module):
    """Frozen decoded merger + latent, with learned spatial/global mixing."""
    def __init__(self, width=512):
        super().__init__()
        self.merger_proj = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, width))
        self.latent_proj = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, width))
        self.position = nn.Parameter(torch.zeros(1, 108, width))
        layer = nn.TransformerEncoderLayer(width, 8, 4*width, dropout=0.,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3*2560))
        self.register_buffer('target_mean', torch.zeros(3, 2560))
        self.register_buffer('target_std', torch.ones(3, 2560))

    def forward(self, latent, merger):
        x = self.merger_proj(merger) + self.latent_proj(latent) + self.position
        return self.head(self.blocks(x)).view(-1, 108, 3, 2560).permute(0, 2, 1, 3)

    def features(self, latent, merger):
        return self(latent, merger).float()*self.target_std[None,:,None,:] + self.target_mean[None,:,None,:]


def inject_actions(policy, template, merger, deep=None):
    """Use the Qwen text backbone directly, preserving visual masks and RoPE."""
    dev = next(policy.parameters()).device
    model = policy.qwen_vl_interface.model
    backbone = model.model
    ids = template['input_ids'].to(dev)
    attention = template['attention_mask'].to(dev)
    grid = template['image_grid_thw'].to(dev)
    embeds = model.get_input_embeddings()(ids)
    merger = merger.reshape(-1, 2560).to(embeds)
    mask, _ = backbone.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=merger)
    embeds = embeds.masked_scatter(mask, merger)
    positions, _ = backbone.get_rope_index(input_ids=ids, image_grid_thw=grid,
                                           video_grid_thw=None, attention_mask=attention)
    deep = None if deep is None else [d.reshape(-1,2560).to(embeds) for d in deep]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = backbone.language_model(inputs_embeds=embeds, attention_mask=attention,
            position_ids=positions, visual_pos_masks=mask[...,0], deepstack_visual_embeds=deep,
            use_cache=False, return_dict=True).last_hidden_state
    queries = policy._gather_action_token_embeddings(hidden, ids, action_token_id=policy.action_token_id)
    with torch.autocast('cuda', dtype=torch.float32):
        return policy.action_model.predict_action(queries).float()


def native_actions(policy, inputs):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = policy.qwen_vl_interface.model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
    queries = policy._gather_action_token_embeddings(hidden, inputs['input_ids'], action_token_id=policy.action_token_id)
    with torch.autocast('cuda', dtype=torch.float32):
        return policy.action_model.predict_action(queries).float()


@torch.no_grad()
def extract(a, policy, codec):
    manifest = []
    split_keys = {}
    for split, limit, frames in [('train', a.train_episodes_per_group, a.train_frames),
                                  ('val', a.val_episodes_per_group, a.val_frames)]:
        ds = RobotwinSVAETokenDataset(a.cache, raw_data_path=a.raw_data_root, split=split,
              num_hist=1, num_pred=1, views=[0], tasks=TASKS)
        groups = defaultdict(list)
        for i, record in enumerate(ds.records):
            groups[(record['task'], record['outcome'])].append(i)
        assert len(groups) == 6, groups.keys()
        selected = []
        for key in sorted(groups):
            candidates = sorted(groups[key], key=lambda i: ds.records[i]['episode_index'])
            random.Random(a.seed).shuffle(candidates)
            assert len(candidates) >= limit
            selected.extend(candidates[:limit])
        split_keys[split] = {(ds.records[i]['dataset'], ds.records[i]['episode_index']) for i in selected}
        if split == 'val':
            assert not split_keys['train'].intersection(split_keys['val'])
        for index in selected:
            record = ds.records[index]
            payload = ds._load_episode(index)
            ids = evenly_spaced_indices(len(payload['actions']), frames)
            images = ds._load_raw_images(index, record, ids)[:,0]
            template = cl.build_policy_template(policy, record['task'], 'success')
            samples = []
            for frame, raw in zip(ids, images):
                pil = Image.fromarray(raw.numpy()).convert('RGB').resize((384,288), Image.Resampling.BILINEAR)
                inputs = policy.qwen_vl_interface.build_qwenvl_inputs(images=[[pil]],
                    instructions=[cl.action_instruction(policy, record['task'], 'success')])
                inputs = {k:v.to(a.device) for k,v in inputs.items() if torch.is_tensor(v)}
                assert torch.equal(inputs['input_ids'], template['input_ids'].to(a.device))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    merged, deep = policy.qwen_vl_interface.model.model.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'])
                merger = torch.cat(list(merged))
                deep = torch.stack(deep)
                assert merger.shape == (108,2560) and deep.shape == (3,108,2560)
                latent = payload['latents']['symmetric_qwen'][frame,0].float().to(a.device)
                decoded = codec.decode(latent[None], [108])[0].reshape(108,2560)
                sample = {'latent':latent.cpu().to(torch.bfloat16), 'merger':decoded.cpu().to(torch.bfloat16),
                          'deep':deep.cpu().to(torch.bfloat16), 'frame':frame}
                if split == 'val':
                    teacher = native_actions(policy, inputs)
                    oracle = inject_actions(policy, template, merger, deep)
                    parity = (oracle-teacher).abs().max().item()
                    # With the same features, dtype and RoPE, numerical parity is mandatory.
                    if parity > 1e-5:
                        raise RuntimeError(f'Native/injected interface parity failed: {parity}')
                    sample.update(teacher=teacher.cpu(), oracle_parity=parity,
                        raw_merger=inject_actions(policy, template, merger).cpu(),
                        codec_merger=inject_actions(policy, template, decoded).cpu(),
                        codec_oracle_deep=inject_actions(policy, template, decoded, deep).cpu(),
                        gt_actions=payload['actions'][frame:frame+50].float())
                samples.append(sample)
            name = f"{split}_{record['dataset']}_{record['episode_index']:06d}.pt"
            torch.save(samples, a.output/'features'/name)
            manifest.append({'file':name, 'split':split, 'task':record['task'], 'outcome':record['outcome'],
                             'dataset':record['dataset'], 'episode_index':record['episode_index'], 'frames':ids})
            event(a.output, phase='extract', split=split, completed_episodes=len(manifest),
                  task=record['task'], episode_index=record['episode_index'])
    save_json(a.output/'manifest.json', {'episodes':manifest, 'split_overlap':0})
    return manifest


def load_samples(a, manifest, split):
    samples, metadata = [], []
    for entry in manifest:
        if entry['split'] == split:
            items = torch.load(a.output/'features'/entry['file'], weights_only=True)
            samples.extend(items)
            metadata.extend([{**entry,'frame':s['frame']} for s in items])
    arrays = {k:torch.stack([s[k] for s in samples]) for k in ['latent','merger','deep']}
    return arrays, samples, metadata


def loss_for(model, batch):
    target = batch['deep'].float()
    normalized = (target-model.target_mean[None,:,None,:])/model.target_std[None,:,None,:]
    pred = model(batch['latent'].float(), batch['merger'].float()).float()
    mse = (pred-normalized).square().mean(dim=(0,2,3))
    restored = pred*model.target_std[None,:,None,:]+model.target_mean[None,:,None,:]
    cosine = F.cosine_similarity(restored, target, dim=-1).mean(dim=(0,2))
    return mse.mean()+0.1*(1-cosine).mean(), mse, cosine


@torch.no_grad()
def validate(model, arrays, a):
    model.eval()
    sums = torch.zeros(7, device=a.device)
    count = len(arrays['latent'])
    for start in range(0,count,a.batch_size):
        b = {k:v[start:start+a.batch_size].to(a.device).float() for k,v in arrays.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss,mse,cos = loss_for(model,b)
        sums += torch.cat([loss[None],mse,cos])*len(b['latent'])
    return (sums/count).cpu().tolist()


@torch.no_grad()
def evaluate_actions(a, policy, model, arrays, samples, metadata, statistics):
    model.eval()
    rows = []
    for i,(s,meta) in enumerate(zip(samples,metadata)):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            deep = model.features(arrays['latent'][i:i+1].to(a.device).float(), arrays['merger'][i:i+1].to(a.device).float())[0]
        pred = inject_actions(policy,cl.build_policy_template(policy,meta['task'],'success'),
                              arrays['merger'][i].to(a.device),deep).cpu()
        mean_actions = inject_actions(policy,cl.build_policy_template(policy,meta['task'],'success'),
                              arrays['merger'][i].to(a.device),model.target_mean[:,None,:].expand(-1,108,-1)).cpu()
        teacher = cl.policy_to_cache_order(s['teacher'][0],statistics)
        row = {k:meta[k] for k in ['task','outcome','dataset','episode_index','frame']}
        row['oracle_parity'] = s['oracle_parity']
        row['metrics'] = {}
        for label,q in [('native_rgb',s['teacher']),('raw_merger',s['raw_merger']),
                        ('codec_merger',s['codec_merger']),('codec_oracle_deep',s['codec_oracle_deep']),('mean_deep',mean_actions),('adapter',pred)]:
            phys = cl.policy_to_cache_order(q[0],statistics)
            j = list(cl.CONTINUOUS_INDICES); g = list(cl.GRIPPER_INDICES)
            valid = min(len(s['gt_actions']),50)
            row['metrics'][label] = {
                'teacher_joint_mae_rad':(phys[:,j]-teacher[:,j]).abs().mean().item(),
                'teacher_gripper_agreement':(phys[:,g]==teacher[:,g]).float().mean().item(),
                'recorded_joint_mae_rad':(phys[:valid,j]-s['gt_actions'][:valid,j]).abs().mean().item()}
        rows.append(row)
        event(a.output,phase='action_eval',completed=i+1,total=len(samples))
    summary = {}
    for task in ['all']+TASKS:
        chosen = [r for r in rows if task=='all' or r['task']==task]
        summary[task] = {label:{k:float(np.mean([r['metrics'][label][k] for r in chosen]))
                         for k in chosen[0]['metrics'][label]} for label in chosen[0]['metrics']}
    save_json(a.output/'action_evaluation.json',{'rows':rows,'summary':summary,
        'scope':'Held-out GT-latent, teacher-forced validation; no World Model rollout or simulator success rate. Failure recorded actions are not success targets.'})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cache','raw-data-root','codec-checkpoint','policy-bundle','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=3047)
    p.add_argument('--train-episodes-per-group',type=int,default=12)
    p.add_argument('--val-episodes-per-group',type=int,default=2)
    p.add_argument('--train-frames',type=int,default=8);p.add_argument('--val-frames',type=int,default=4)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--validate-every',type=int,default=100);p.add_argument('--lr',type=float,default=3e-4)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'features').mkdir()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(tasks=TASKS,loss='train-channel-normalized MSE + 0.1 cosine distance',
        frozen=['S-VAE','World Model (not loaded)','policy'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input='GT cached cam_high posterior mean and frozen decoded merger',target='Qwen post-merger DeepStack layers 5,11,17',
        selection='best held-out feature loss; action results are validation, not independent test')
    save_json(a.output/'config.json',config)
    (a.output/'train_deepstack_adapter.snapshot.py').write_bytes(Path(__file__).read_bytes())
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(8);torch.cuda.set_device(a.device)
    event(a.output,phase='load_models')
    policy,_,_=cl.load_policy(a.policy_bundle,torch.device(a.device))
    codec,_=load_codec_checkpoint(a.codec_checkpoint,device=torch.device(a.device))
    codec.eval().requires_grad_(False)
    assert not any(p.requires_grad for p in policy.parameters())
    manifest=extract(a,policy,codec)
    train,_,_=load_samples(a,manifest,'train')
    val,val_samples,val_meta=load_samples(a,manifest,'val')
    model=DeepStackAdapter().to(a.device)
    # Streaming train-only moments avoid a large temporary float32 cache.
    total=torch.zeros(3,2560,dtype=torch.float64);total2=total.clone();count=0
    for chunk in train['deep'].split(16):
        d=chunk.double();total+=d.sum((0,2));total2+=d.square().sum((0,2));count+=len(d)*108
    mean=total/count;std=(total2/count-mean.square()).clamp_min(1e-6).sqrt()
    model.target_mean.copy_(mean.float());model.target_std.copy_(std.float().clamp_min(.01))
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    initial=validate(model,val,a)
    event(a.output,phase='training_start',train_frames=len(train['latent']),val_frames=len(val['latent']),
          parameters=sum(p.numel() for p in model.parameters()),val_loss=initial[0],oracle_parity_max=max(s['oracle_parity'] for s in val_samples))
    best=float('inf');history=[];gen=torch.Generator().manual_seed(a.seed);order=torch.empty(0,dtype=torch.long);cursor=0
    for step in range(1,a.steps+1):
        if cursor+a.batch_size>len(order):
            order=torch.randperm(len(train['latent']),generator=gen);cursor=0
        ids=order[cursor:cursor+a.batch_size];cursor+=a.batch_size
        batch={k:v[ids].to(a.device).float() for k,v in train.items()}
        model.train();optimizer.zero_grad(set_to_none=True)
        lr=a.lr*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/a.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=loss_for(model,batch)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%25==0:event(a.output,phase='train',step=step,total_steps=a.steps,loss=loss.item(),grad_norm=grad.item(),lr=lr)
        if step==1 or step%a.validate_every==0 or step==a.steps:
            values=validate(model,val,a);history.append({'step':step,'loss':values[0],'mse':values[1:4],'cosine':values[4:]})
            event(a.output,phase='validation',**history[-1])
            if values[0]<best:
                best=values[0]
                torch.save({'model':model.state_dict(),'step':step,'validation':history[-1],'config':config},a.output/'adapter_best.pt')
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':config},a.output/'adapter_last.pt')
            save_json(a.output/'feature_evaluation.json',{'initial':initial,'history':history})
    state=torch.load(a.output/'adapter_best.pt',map_location=a.device,weights_only=False);model.load_state_dict(state['model'])
    statistics=json.loads((a.policy_bundle/'model/dataset_statistics.json').read_text())
    summary=evaluate_actions(a,policy,model,val,val_samples,val_meta,statistics)
    event(a.output,phase='complete',best_step=state['step'],best_val_loss=best,actions=summary)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        import sys
        if '--output' in sys.argv:
            out = Path(sys.argv[sys.argv.index('--output')+1])
            if out.is_dir() and not isinstance(exc, FileExistsError):
                event(out, phase='failed', error=repr(exc))
        raise
