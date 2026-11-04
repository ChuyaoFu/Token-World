#!/usr/bin/env python3
"""All-task cam_high adapter pipeline. Sharded export -> streamed train -> eval -> owned-cache cleanup."""
import argparse
from collections import OrderedDict, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import uuid

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tokenworld.codec.common import load_codec_checkpoint, evenly_spaced_indices
from tokenworld.codec.deepstack_adapter import DeepStackAdapter, inject_actions, native_actions
from tokenworld.data.dataset import RobotwinSVAETokenDataset
from tokenworld.evaluation import closed_loop as cl


def dump(path, value):
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2)); temp.replace(path)


def save(path, value):
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    torch.save(value, temp); temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def log(a, phase, **kw):
    row = dict(phase=phase, time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **kw)
    print(json.dumps(row),flush=True)
    with (a.output/f'metrics.{a.stage}.rank{a.rank:02d}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')


def contract(a):
    names = ['cache','raw_data_root','codec_checkpoint','policy_bundle','seed','train_frames','val_frames',
             'tasks','max_episodes_per_group','expected_tasks','epochs','steps','batch_size','lr','validate_every','action_every']
    data = {k:str(getattr(a,k)) if isinstance(getattr(a,k),Path) else getattr(a,k) for k in names}
    ckpts = sorted((a.policy_bundle/'model/checkpoints').glob('*.pt'))
    paths = [a.codec_checkpoint, a.cache/'index.json', a.cache/'cache_spec.json',
             a.policy_bundle/'model/config.full.yaml', a.policy_bundle/'model/dataset_statistics.json', *ckpts]
    data['asset_identity'] = {str(p.resolve()):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths}
    data['source_sha256'] = digest(Path(__file__))
    import tokenworld.codec.deepstack_adapter as adapter_module
    data['adapter_source_sha256'] = digest(Path(adapter_module.__file__))
    return data


def prepare(a):
    requested = contract(a)
    if (a.output/'run.json').exists():
        run=json.loads((a.output/'run.json').read_text())
        if run['contract'] != requested:raise ValueError('Existing run contract differs; use a fresh output or the original arguments')
        print('Reusing matching run',run['run_id'],flush=True);return
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('New output must be empty')
    records=json.loads((a.cache/'index.json').read_text())['records']
    groups=defaultdict(list)
    for r in records:
        if a.tasks and r['task'] not in a.tasks:continue
        if r['split'] in ['train','val']:groups[(r['split'],r['task'],r['outcome'])].append(r)
    tasks=sorted({k[1] for k in groups})
    if len(tasks)!=a.expected_tasks:raise ValueError(f'Expected {a.expected_tasks} tasks, found {len(tasks)}')
    selected=[];split_keys=defaultdict(set)
    for key,entries in sorted(groups.items()):
        entries=sorted(entries,key=lambda r:(r['dataset'],r['episode_index']))
        if a.max_episodes_per_group:
            random.Random(a.seed).shuffle(entries);entries=entries[:a.max_episodes_per_group]
        for r in entries:
            ids=evenly_spaced_indices(r['num_frames'],a.train_frames if r['split']=='train' else a.val_frames)
            row={k:r[k] for k in ['dataset','episode_index','task','outcome','split','path']}
            row.update(frames=ids,file=f"{r['split']}_{r['dataset']}_{r['episode_index']:06d}.pt")
            if (r['dataset'],r['episode_index']) in split_keys[r['split']]:raise ValueError('Duplicate episode')
            split_keys[r['split']].add((r['dataset'],r['episode_index']));selected.append(row)
    if split_keys['train'] & split_keys['val']:raise ValueError('Episode leakage')
    if not split_keys['train'] or not split_keys['val']:raise ValueError('Empty split')
    counts={s:sum(len(r['frames']) for r in selected if r['split']==s) for s in ['train','val']}
    # bf16 latent + frozen merger + three DeepStack outputs, plus metadata margin.
    estimated=sum(counts.values())*2*(108*16+4*108*2560)+len(selected)*160000
    required=int(estimated*1.15)+20*1024**3
    free=shutil.disk_usage(a.output).free
    if free<required:raise RuntimeError(f'Insufficient free storage: need {required/1024**3:.1f} GiB, have {free/1024**3:.1f} GiB')
    run=dict(run_id=str(uuid.uuid4()),contract=requested,tasks=tasks,counts=counts,
             episodes={s:len(v) for s,v in split_keys.items()},estimated_feature_bytes=estimated)
    dump(a.output/'manifest.json',{'episodes':selected,'split_overlap':0})
    (a.output/'features').mkdir()
    dump(a.output/'features/.owner.json',{'run_id':run['run_id'],'output':str(a.output.resolve())})
    (a.output/'deepstack_pipeline.snapshot.py').write_bytes(Path(__file__).read_bytes())
    import tokenworld.codec.deepstack_adapter as adapter_module
    (a.output/'deepstack_adapter.snapshot.py').write_bytes(Path(adapter_module.__file__).read_bytes())
    dump(a.output/'run.json',run)
    log(a,'prepared',**run)


def context(a):
    run=json.loads((a.output/'run.json').read_text())
    if run['contract']!=contract(a):raise ValueError('Run arguments/assets/code changed; refusing to mix artifacts')
    entries=json.loads((a.output/'manifest.json').read_text())['episodes']
    return run,entries


def feature_path(a,entry):
    name=entry['file']
    if Path(name).name!=name or not name.endswith('.pt'):raise ValueError('Unsafe feature filename')
    return a.output/'features'/name


def read_feature(a,entry,run):
    path=feature_path(a,entry)
    if path.is_symlink():raise ValueError('Symlink feature forbidden')
    value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if value['run_id']!=run['run_id'] or value['entry']!=entry:raise ValueError('Feature provenance mismatch')
    n=len(entry['frames'])
    for key,shape in [('latent',(n,108,16)),('merger',(n,108,2560)),('deep',(n,3,108,2560))]:
        if tuple(value[key].shape)!=shape or value[key].dtype!=torch.bfloat16:raise ValueError(f'Invalid {key} shape/dtype')
    if entry['split']=='val' and len(value['actions'])!=n:raise ValueError('Missing validation action targets')
    return value


@torch.no_grad()
def export(a):
    run,entries=context(a)
    if (a.output/'cleanup.json').exists():
        print('Run already cleaned after success; export skipped',flush=True);return
    torch.cuda.set_device(a.device)
    policy,_,_=cl.load_policy(a.policy_bundle,torch.device(a.device))
    codec,_=load_codec_checkpoint(a.codec_checkpoint,device=torch.device(a.device));codec.eval().requires_grad_(False)
    datasets={s:RobotwinSVAETokenDataset(a.cache,raw_data_path=a.raw_data_root,split=s,num_hist=1,num_pred=1,views=[0],tasks=a.tasks or None) for s in ['train','val']}
    lookups={s:{(r['dataset'],r['episode_index']):i for i,r in enumerate(ds.records)} for s,ds in datasets.items()}
    assigned=entries[a.rank::a.world_size]
    for index,entry in enumerate(assigned):
        dest=feature_path(a,entry)
        if dest.exists():read_feature(a,entry,run);continue
        ds=datasets[entry['split']];ri=lookups[entry['split']][(entry['dataset'],entry['episode_index'])]
        payload=ds._load_episode(ri);images=ds._load_raw_images(ri,ds.records[ri],entry['frames'])[:,0]
        template=cl.build_policy_template(policy,entry['task'],'success')
        latents=[];mergers=[];targets=[];actions=[]
        for frame,raw in zip(entry['frames'],images):
            pil=Image.fromarray(raw.numpy()).convert('RGB').resize((384,288),Image.Resampling.BILINEAR)
            inputs=policy.qwen_vl_interface.build_qwenvl_inputs(images=[[pil]],instructions=[cl.action_instruction(policy,entry['task'],'success')])
            inputs={k:v.to(a.device) for k,v in inputs.items() if torch.is_tensor(v)}
            if not torch.equal(inputs['input_ids'],template['input_ids'].to(a.device)):raise ValueError('Template mismatch')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                raw_merger,deep=policy.qwen_vl_interface.model.model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'])
            raw_merger=torch.cat(list(raw_merger));deep=torch.stack(deep)
            latent=payload['latents']['symmetric_qwen'][frame,0].float().to(a.device)
            merger=codec.decode(latent[None],[108])[0].reshape(108,2560)
            if any(not torch.isfinite(v).all() for v in [latent,merger,deep]):raise ValueError('Nonfinite feature')
            latents.append(latent.cpu().to(torch.bfloat16));mergers.append(merger.cpu().to(torch.bfloat16));targets.append(deep.cpu().to(torch.bfloat16))
            if entry['split']=='val':
                teacher=native_actions(policy,inputs)
                oracle=inject_actions(policy,template,raw_merger,deep)
                parity=(oracle-teacher).abs().max().item()
                if parity>1e-5:raise ValueError(f'Native feature injection parity failed: {parity}')
                actions.append({'teacher':teacher.cpu(),'oracle_parity':parity,
                    'codec_merger':inject_actions(policy,template,merger).cpu(),
                    'codec_oracle_deep':inject_actions(policy,template,merger,deep).cpu()})
        deep=torch.stack(targets)
        value=dict(run_id=run['run_id'],entry=entry,latent=torch.stack(latents),merger=torch.stack(mergers),deep=deep,actions=actions)
        if entry['split']=='train':
            d=deep.double();value.update(sum=d.sum((0,2)),sum2=d.square().sum((0,2)),count=len(d)*108)
        save(dest,value)
        if index%10==0 or index+1==len(assigned):log(a,'export',completed=index+1,total=len(assigned))
    dump(a.output/f'export.rank{a.rank:02d}.json',dict(status='complete',run_id=run['run_id'],episodes=len(assigned),world_size=a.world_size))


def finalize(a):
    run,entries=context(a)
    if (a.output/'cleanup.json').exists():return
    total=torch.zeros(3,2560,dtype=torch.float64);total2=total.clone();count=0;parity=0.;files=[]
    for entry in entries:
        v=read_feature(a,entry,run)
        if entry['split']=='train':total+=v['sum'];total2+=v['sum2'];count+=v['count']
        else:parity=max(parity,max(x['oracle_parity'] for x in v['actions']))
        files.append({'file':entry['file'],'bytes':feature_path(a,entry).stat().st_size})
    mean=total/count;std=(total2/count-mean.square()).clamp_min(1e-6).sqrt().float().clamp_min(.01)
    save(a.output/'target_stats.pt',dict(mean=mean.float(),std=std,count=count,run_id=run['run_id']))
    dump(a.output/'export_complete.json',dict(status='complete',run_id=run['run_id'],files=files,max_oracle_parity=parity))
    log(a,'export_verified',episodes=len(entries),max_oracle_parity=parity)


class FeatureDataset(Dataset):
    def __init__(self,a,entries,run,split):
        self.a=a;self.run=run;self.entries=[e for e in entries if e['split']==split]
        self.samples=[(i,j) for i,e in enumerate(self.entries) for j in range(len(e['frames']))]
        self.cache=OrderedDict()
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        i,j=self.samples[index]
        if i not in self.cache:
            self.cache[i]=read_feature(self.a,self.entries[i],self.run)
            while len(self.cache)>32:self.cache.popitem(last=False)
        self.cache.move_to_end(i);v=self.cache[i]
        return {k:v[k][j] for k in ['latent','merger','deep']}


class StepBatches(Sampler):
    """Deterministic epoch permutations, including resume at an optimizer step."""
    def __init__(self,n,batch,start,end,seed):self.n=n;self.batch=batch;self.start=start;self.end=end;self.seed=seed
    def __len__(self):return self.end-self.start
    def __iter__(self):
        per_epoch=self.n//self.batch
        if per_epoch<1:raise ValueError('Training split smaller than batch size')
        old_epoch=-1
        for step in range(self.start,self.end):
            epoch,offset=divmod(step,per_epoch)
            if epoch!=old_epoch:
                order=torch.randperm(self.n,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist();old_epoch=epoch
            yield order[offset*self.batch:(offset+1)*self.batch]


def feature_loss(model,b):
    target=b['deep'].float();mu=model.target_mean[None,:,None,:];std=model.target_std[None,:,None,:]
    pred=model(b['latent'].float(),b['merger'].float()).float()
    mse=(pred-(target-mu)/std).square().mean((0,2,3))
    cos=torch.nn.functional.cosine_similarity(pred*std+mu,target,dim=-1).mean((0,2))
    return mse.mean()+.1*(1-cos).mean(),mse,cos


@torch.no_grad()
def validate(a,model,dataset):
    model.eval();total=torch.zeros(7,device=a.device);count=0
    for b in DataLoader(dataset,batch_size=a.batch_size,num_workers=0):
        b={k:v.to(a.device).float() for k,v in b.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,mse,cos=feature_loss(model,b)
        total+=torch.cat([loss[None],mse,cos])*len(b['latent']);count+=len(b['latent'])
    values=(total/count).cpu().tolist()
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite validation')
    return dict(loss=values[0],mse=values[1:4],cosine=values[4:])


@torch.no_grad()
def action_eval(a,model,policy,entries,run,monitor=False):
    model.eval();statistics=json.loads((a.policy_bundle/'model/dataset_statistics.json').read_text())
    rows=[];seen=set()
    for entry in entries:
        if entry['split']!='val':continue
        group=(entry['task'],entry['outcome'])
        if monitor and group in seen:continue
        seen.add(group);v=read_feature(a,entry,run);template=cl.build_policy_template(policy,entry['task'],'success')
        for i,frame in enumerate(entry['frames'][:1] if monitor else entry['frames']):
            latent=v['latent'][i:i+1].to(a.device).float();merger=v['merger'][i:i+1].to(a.device).float()
            with torch.autocast('cuda',dtype=torch.bfloat16):deep=model.features(latent,merger)[0]
            q=inject_actions(policy,template,merger,deep).cpu()[0]
            teacher=cl.policy_to_cache_order(v['actions'][i]['teacher'][0],statistics)
            row={k:entry[k] for k in ['task','outcome','dataset','episode_index']};row.update(frame=frame,metrics={})
            for label,act in [('adapter',q),('codec_merger',v['actions'][i]['codec_merger'][0]),('codec_oracle_deep',v['actions'][i]['codec_oracle_deep'][0])]:
                phys=cl.policy_to_cache_order(act,statistics);j=list(cl.CONTINUOUS_INDICES);g=list(cl.GRIPPER_INDICES)
                row['metrics'][label]={'teacher_joint_mae_rad':(phys[:,j]-teacher[:,j]).abs().mean().item(),
                                       'teacher_gripper_agreement':(phys[:,g]==teacher[:,g]).float().mean().item()}
            if not all(math.isfinite(x) for m in row['metrics'].values() for x in m.values()):raise ValueError('Nonfinite action metric')
            rows.append(row)
    summary={}
    for task in ['all']+run['tasks']:
        chosen=[r for r in rows if task=='all' or r['task']==task]
        summary[task]={label:{k:float(np.mean([r['metrics'][label][k] for r in chosen])) for k in chosen[0]['metrics'][label]} for label in chosen[0]['metrics']}
    return dict(rows=rows,summary=summary,scope='GT-latent teacher agreement; validation, not independent test or closed-loop success')


def train(a):
    run,entries=context(a)
    if (a.output/'training_complete.json').exists():print('Training already complete',flush=True);return
    if not (a.output/'export_complete.json').exists():raise ValueError('Finalize export before training')
    torch.cuda.set_device(a.device);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    tr=FeatureDataset(a,entries,run,'train');val=FeatureDataset(a,entries,run,'val')
    steps=a.steps or a.epochs*(len(tr)//a.batch_size)
    if steps<1:raise ValueError('Empty training schedule')
    model=DeepStackAdapter().to(a.device);stats=torch.load(a.output/'target_stats.pt',weights_only=True)
    model.target_mean.copy_(stats['mean']);model.target_std.copy_(stats['std'])
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    start=0;best=float('inf');history=[];action_history=[]
    if (a.output/'adapter_last.pt').exists():
        checkpoint=torch.load(a.output/'adapter_last.pt',map_location=a.device,weights_only=False)
        if checkpoint['run_id']!=run['run_id'] or checkpoint['total_steps']!=steps:raise ValueError('Resume contract mismatch')
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer']);start=checkpoint['step'];best=checkpoint['best']
        history=checkpoint['history'];action_history=checkpoint['action_history']
    policy,_,_=cl.load_policy(a.policy_bundle,torch.device(a.device))
    loader=DataLoader(tr,batch_sampler=StepBatches(len(tr),a.batch_size,start,steps,a.seed),num_workers=a.workers,
                      **({'persistent_workers':True,'multiprocessing_context':'spawn','prefetch_factor':2} if a.workers else {}))
    log(a,'training_start',step=start,total_steps=steps,train_frames=len(tr),validation_frames=len(val))
    for step,b in enumerate(loader,start+1):
        model.train();b={k:v.to(a.device).float() for k,v in b.items()};opt.zero_grad(set_to_none=True)
        lr=a.lr*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/steps)))
        for g in opt.param_groups:g['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=feature_loss(model,b)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%100==0:log(a,'train',step=step,total_steps=steps,loss=loss.item(),grad_norm=grad.item(),lr=lr)
        check=step==1 or step%a.validate_every==0 or step==steps
        action_check=step==1 or step%a.action_every==0 or step==steps
        if check:
            metrics=validate(a,model,val);history.append(dict(step=step,**metrics));log(a,'validation',step=step,**metrics)
            if metrics['loss']<best:
                best=metrics['loss'];save(a.output/'adapter_best.pt',dict(model=model.state_dict(),step=step,run_id=run['run_id'],validation=metrics,config=run))
        if action_check:
            result=action_eval(a,model,policy,entries,run,monitor=True)
            action_history.append(dict(step=step,summary=result['summary']));dump(a.output/'action_monitor.json',action_history)
            log(a,'action_monitor',step=step,summary=result['summary']['all'])
        if check or action_check:
            save(a.output/'adapter_last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,total_steps=steps,
                 best=best,history=history,action_history=action_history,run_id=run['run_id'],config=run))
            dump(a.output/'feature_evaluation.json',history)
    # The worker iterator is exhausted; drop mmap references before subsequent cleanup.
    del loader,tr,val
    state=torch.load(a.output/'adapter_best.pt',map_location=a.device,weights_only=False);model.load_state_dict(state['model'])
    result=action_eval(a,model,policy,entries,run)
    result.update(status='complete',run_id=run['run_id'],checkpoint_sha256=digest(a.output/'adapter_best.pt'),best_step=state['step'])
    dump(a.output/'action_evaluation.json',result)
    complete=dict(status='complete',run_id=run['run_id'],step=steps,best_step=state['step'],best_validation=state['validation'],
                  best_sha256=digest(a.output/'adapter_best.pt'),last_sha256=digest(a.output/'adapter_last.pt'))
    dump(a.output/'training_complete.json',complete);log(a,'complete',**complete)


def cleanup(a):
    run,entries=context(a)
    marker=a.output/'cleanup.json'
    if marker.exists():print('Owned feature cleanup already complete',flush=True);return
    complete=json.loads((a.output/'training_complete.json').read_text())
    evaluation=json.loads((a.output/'action_evaluation.json').read_text())
    if any(x['status']!='complete' or x['run_id']!=run['run_id'] for x in [complete,evaluation]):raise ValueError('Training/evaluation not complete')
    if digest(a.output/'adapter_best.pt')!=complete['best_sha256'] or evaluation['checkpoint_sha256']!=complete['best_sha256']:raise ValueError('Best checkpoint integrity mismatch')
    if digest(a.output/'adapter_last.pt')!=complete['last_sha256']:raise ValueError('Last checkpoint integrity mismatch')
    for name in ['adapter_best.pt','adapter_last.pt']:
        checkpoint=torch.load(a.output/name,map_location='cpu',weights_only=True,mmap=True)
        if checkpoint['run_id']!=run['run_id'] or any(not torch.isfinite(t).all() for t in checkpoint['model'].values()):raise ValueError('Invalid saved model')
    feature_dir=a.output/'features'
    if feature_dir.is_symlink() or feature_dir.resolve().parent!=a.output.resolve():raise ValueError('Unsafe feature directory')
    owner=json.loads((feature_dir/'.owner.json').read_text())
    if owner!={'run_id':run['run_id'],'output':str(a.output.resolve())}:raise ValueError('Feature directory not owned by this run')
    names={e['file'] for e in entries}
    existing=list(feature_dir.iterdir())
    if any(p.is_symlink() or not p.is_file() or (p.name not in names and p.name!='.owner.json') for p in existing):raise ValueError('Unexpected files: refusing cleanup')
    sizes={r['file']:r['bytes'] for r in json.loads((a.output/'export_complete.json').read_text())['files']}
    deleted=0;freed=0
    for name in sorted(names):
        path=feature_dir/name
        if path.exists():
            if path.stat().st_size!=sizes[name]:raise ValueError('Feature file changed: '+name)
            size=path.stat().st_size;path.unlink();deleted+=1;freed+=size
            with (a.output/'cleanup_actions.jsonl').open('a') as f:f.write(json.dumps({'file':name,'bytes':size})+'\n')
    # Keep only the tiny ownership marker to allow interrupted cleanup to resume safely.
    dump(marker,dict(status='complete',run_id=run['run_id'],deleted_this_attempt=deleted,bytes_this_attempt=freed,
                     scope='Only feature files explicitly listed in this run manifest; owner marker retained'))
    log(a,'cleanup_complete',deleted=deleted,bytes=freed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','export','finalize','train','cleanup'],required=True)
    for name in ['cache','raw-data-root','codec-checkpoint','policy-bundle','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tasks',nargs='*');p.add_argument('--expected-tasks',type=int,default=50)
    p.add_argument('--max-episodes-per-group',type=int,default=0)
    p.add_argument('--train-frames',type=int,default=8);p.add_argument('--val-frames',type=int,default=4)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--steps',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--validate-every',type=int,default=1000);p.add_argument('--action-every',type=int,default=2000)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=3047)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.output=a.output.resolve()
    for key in ['cache','raw_data_root','codec_checkpoint','policy_bundle']:setattr(a,key,getattr(a,key).resolve())
    a.rank=int(os.environ.get('RANK',0));a.world_size=int(os.environ.get('WORLD_SIZE',1)) if a.stage=='export' else 1
    if a.stage=='export':a.device=f"cuda:{int(os.environ.get('LOCAL_RANK',0))}"
    elif a.rank!=0:raise ValueError('Only export is multi-process')
    for key in ['train_frames','val_frames','epochs','batch_size','expected_tasks','validate_every','action_every']:
        if getattr(a,key)<=0:raise ValueError(key+' must be positive')
    if a.steps<0 or a.workers<0 or a.max_episodes_per_group<0:raise ValueError('Negative runtime option')
    torch.set_num_threads(4)
    globals()[a.stage](a)


if __name__=='__main__':main()
