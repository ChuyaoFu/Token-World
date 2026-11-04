#!/usr/bin/env python3
"""Warm-start shape/perceptual decoder fine-tuning; PPU single-card or DDP.

--train-probes: fast, immutable pilot (optionally mix raw merger supervision).
--cache/--data-root: full split, random frames, on-the-fly frozen codec decode;
no huge expanded-feature cache and no full-Qwen inference during training.
"""
import argparse,json,os,sys,time,random
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader,DistributedSampler
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/table1_robotwin')]
from decoder_improvement.models import load_decoder
from tokenworld.decoder.metrics.lpips.modules.lpips import LPIPS
from tokenworld.codec.common import load_codec_checkpoint
import evaluate as table_eval
from diagnose import sha


class CachedFrames(Dataset):
    def __init__(self,cache,data_root,frames):
        self.cache=cache;self.frames=frames
        self.records=[r for r in json.loads((cache/'index.json').read_text())['records'] if r['split']=='train']
        self.raw=table_eval.RawCacheDatasetAdapter(json.loads((cache/'cache_spec.json').read_text()),data_root)
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i];p=table_eval._torch_load(self.cache/r['path']);n=min(r['num_frames'],len(p['frame_indices']))
        ids=sorted(random.sample(range(n),min(n,self.frames)))
        rgb=torch.cat([table_eval.raw_cam_high_rgb(self.raw,r,p,j,j+1) for j in ids])
        return p['latents']['symmetric_qwen'][ids,0].float(),F.interpolate(rgb,size=(224,224),mode='bilinear',align_corners=False)


def edge(x):return [x[...,1:,:]-x[...,:-1,:],x[...,:,1:]-x[...,:,:-1]]
def losses(pred,target,net,a):
    x=pred.float();y=target.float();mse=F.mse_loss(x,y);l1=F.l1_loss(x,y)
    e=sum(F.l1_loss(u,v) for u,v in zip(edge(x),edge(y)))/2
    percept=net(x,y)
    # Emphasize true boundaries using weights from GT only, with bounded gain.
    lum=y.mean(1,keepdim=True);dx=F.pad((lum[:,:,:,1:]-lum[:,:,:,:-1]).abs(),(0,1));dy=F.pad((lum[:,:,1:,:]-lum[:,:,:-1,:]).abs(),(0,0,0,1))
    weight=1+4*(dx+dy).clamp(max=.5)
    boundary=((x-y).abs()*weight).mean()
    contour=x.new_zeros(());foreground=x.new_zeros(())
    if a.color_shape_weight > 0:
        def colored(q):
            other=torch.maximum(q.roll(1,1),q.roll(2,1))
            return ((q-other-.25)/.5).clamp(0,1)
        px,py=colored(x),colored(y)
        mass=py.sum((2,3));valid=mass>8
        dice=1-(2*(px*py).sum((2,3))+1)/(px.square().sum((2,3))+py.square().sum((2,3))+1)
        contour=(dice*valid).sum()/valid.sum().clamp_min(1)
        foreground=((x-y).abs()*py).sum((2,3))/mass.clamp_min(1)
        foreground=(foreground*valid).sum()/valid.sum().clamp_min(1)
    total=mse+a.l1_weight*l1+a.lpips_weight*percept+a.edge_weight*e+a.boundary_weight*boundary+a.color_shape_weight*(contour+foreground)
    return total,dict(mse=mse.item(),l1=l1.item(),lpips=percept.item(),edge=e.item(),boundary=boundary.item(),color_contour=contour.item(),foreground_l1=foreground.item())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--architecture',choices=['baseline','multiscale'],default='baseline');p.add_argument('--output',type=Path,required=True);p.add_argument('--train-probes',type=Path);p.add_argument('--supplement-probes',type=Path);p.add_argument('--supplement-fraction',type=float,default=.25);p.add_argument('--validation-probes',type=Path,required=True);p.add_argument('--cache',type=Path);p.add_argument('--data-root',type=Path);p.add_argument('--codec',type=Path,default=ROOT/'outputs/svae_d16_3views/best.pt');p.add_argument('--steps',type=int,default=120);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--frames-per-episode',type=int,default=2);p.add_argument('--workers',type=int,default=2);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--lpips-weight',type=float,default=.05);p.add_argument('--edge-weight',type=float,default=.1);p.add_argument('--l1-weight',type=float,default=.1);p.add_argument('--boundary-weight',type=float,default=.1);p.add_argument('--color-shape-weight',type=float,default=0.0);p.add_argument('--raw-fraction',type=float,default=.25);p.add_argument('--checkpoint-every',type=int,default=0);p.add_argument('--device',default='cuda:1');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));dev=f'cuda:{local}' if world>1 else a.device
    torch.set_num_threads(4);torch.cuda.set_device(dev);torch.manual_seed(a.seed+rank);random.seed(a.seed+rank)
    if world>1:dist.init_process_group('nccl')
    if rank==0:
        a.output.mkdir(parents=True,exist_ok=True)
        if (a.output/'decoder_candidate.pt').exists():raise FileExistsError(a.output/'decoder_candidate.pt')
    if world>1:dist.barrier(device_ids=[local])
    os.environ.setdefault('TORCH_HOME',str(ROOT/'outputs/pretrained/torch'))
    model=load_decoder(str(a.baseline),dev,architecture=a.architecture);model.train()
    if world>1:model=DDP(model,device_ids=[local])
    net=LPIPS('alex','0.1').to(dev).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps,eta_min=a.lr*.1)
    validation=torch.load(a.validation_probes,map_location='cpu',weights_only=False)
    valkeys={(c['dataset'],c['episode_index']) for c in validation if c['split']=='val'}
    codec=None
    supplements=None
    if a.train_probes:
        probes=[c for c in torch.load(a.train_probes,map_location='cpu',weights_only=False) if c['split']=='train']
        assert probes and not valkeys.intersection((c['dataset'],c['episode_index']) for c in probes)
        features={k:torch.stack([c[k] for c in probes]) for k in ['raw','cached']}
        target=F.interpolate(torch.stack([c['target'] for c in probes]),size=(224,224),mode='bilinear',align_corners=False)
        epoch_iter=None
    else:
        assert a.cache and a.data_root
        ds=CachedFrames(a.cache,a.data_root,a.frames_per_episode);assert not valkeys.intersection((r['dataset'],r['episode_index']) for r in ds.records)
        sampler=DistributedSampler(ds,world,rank,shuffle=True,seed=a.seed) if world>1 else None
        loader=DataLoader(ds,batch_size=a.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=a.workers,drop_last=True,pin_memory=True,multiprocessing_context='spawn' if a.workers else None,persistent_workers=a.workers>0)
        epoch_iter=iter(loader);epoch=0
        codec,_=load_codec_checkpoint(a.codec,device=torch.device(dev));codec.eval().requires_grad_(False)
    if a.supplement_probes:
        assert not a.train_probes, 'supplements are for full-cache training'
        supplements=[c for c in torch.load(a.supplement_probes,map_location='cpu',weights_only=False) if c['split']=='train' and c['dataset']=='sim_cabinet']
        assert supplements and not valkeys.intersection((c['dataset'],c['episode_index']) for c in supplements)
        supplement_features={k:torch.stack([c[k] for c in supplements]) for k in ['raw','cached']}
        supplement_target=F.interpolate(torch.stack([c['target'] for c in supplements]),size=(224,224),mode='bilinear',align_corners=False)
    config=torch.load(a.baseline,map_location='cpu',weights_only=False,mmap=True)['config']
    provenance={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'baseline_sha256':sha(a.baseline),'validation_sha256':sha(a.validation_probes),'world_size':world,'train_scope':'immutable pilot' if a.train_probes else 'all train episodes; random frames; frozen S-VAE; no val episodes','selection':'fixed training steps; validation never backpropagated; final checkpoint, no best-on-probe selection'}
    if rank==0:(a.output/'training_protocol.json').write_text(json.dumps(provenance,indent=2))
    sample_rng=torch.Generator().manual_seed(a.seed+rank)
    history=[];start=time.monotonic()
    def save(step,name):
        if rank==0:
            m=model.module if world>1 else model
            torch.save(dict(decoder_state_dict=m.state_dict(),decoder_architecture=a.architecture,config=config,step=step,provenance=provenance),a.output/name)
    for step in range(1,a.steps+1):
        if a.train_probes:
            ids=torch.randint(len(probes),(a.batch_size,),generator=sample_rng);kind='raw' if random.random()<a.raw_fraction else 'cached';x=features[kind][ids].to(dev);y=target[ids].to(dev)
        elif supplements and random.Random(a.seed+step).random()<a.supplement_fraction:
            ids=torch.randint(len(supplements),(a.batch_size*a.frames_per_episode,),generator=sample_rng);kind='raw' if random.random()<a.raw_fraction else 'cached';x=supplement_features[kind][ids].to(dev);y=supplement_target[ids].to(dev)
        else:
            try:z,y=next(epoch_iter)
            except StopIteration:
                epoch+=1
                if sampler is not None:sampler.set_epoch(epoch)
                epoch_iter=iter(loader);z,y=next(epoch_iter)
            z=z.reshape(-1,108,16).to(dev);y=y.reshape(-1,3,224,224).to(dev)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):x=torch.stack(codec.decode(z,[108]*len(z)))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x)[0]
        loss,vals=losses(pred,y,net,a)
        if not torch.isfinite(loss):raise FloatingPointError(vals)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step()
        if step%10==0 or step==1:
            row=dict(step=step,loss=loss.item(),**vals,elapsed=time.monotonic()-start,lr=scheduler.get_last_lr()[0]);history.append(row)
            if rank==0:print(json.dumps(row),flush=True)
        if a.checkpoint_every and step%a.checkpoint_every==0:save(step,f'decoder_step_{step:06d}.pt')
    save(a.steps,'decoder_candidate.pt')
    if rank==0:(a.output/'history.json').write_text(json.dumps(history,indent=2));print('COMPLETE',flush=True)
    if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
