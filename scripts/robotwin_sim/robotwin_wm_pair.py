"""Paired physical RoboTwin scoring with first-frame-only conditioning of TokenWorld."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image
from tokenworld.codec.common import load_codec_checkpoint
from tokenworld.codec.deepstack_adapter import DeepStackAdapter, adapter_policy_actions
from tokenworld.decoder.checkpoint import load_external_decoder
from tokenworld.evaluation import closed_loop as cl
from tokenworld.evaluation.rollout import WorldModelRollout

TASKS = ('put_object_cabinet','shake_bottle_horizontally','turn_switch')
HERE = Path(__file__).resolve().parent

class EpisodeStream:
    def __init__(self, models, cache_mode="gru"):
        if cache_mode not in ("gru", "full"):
            raise ValueError(f"Unsupported recurrent stream mode: {cache_mode}")
        self.cache_mode = cache_mode
        self.models = models
        self.cache = None
        self.steps = 0
        self.seconds = 0.
        self.kv_bytes = 0

    @torch.inference_mode()
    def advance(self, state, proprio, context, seed, n):
        if len(context) != n + 1:
            raise ValueError('Expected the boundary action followed by exactly n executed targets')
        model = self.models
        d, c = cl.action_to_model_inputs(context, model.action_mean, model.action_std)
        torch.cuda.synchronize(model.dev)
        start = time.perf_counter()
        torch.manual_seed(seed)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model.wm.rollout(state, d[None], c[None], proprio,
                                      time_cache=self.cache, return_time_cache=True,
                                      gru_only_boundary=self.cache_mode == "gru")
        self.cache = result.time_cache
        self.steps += n
        kv = self.cache.next_kv_cache
        valid_kv = kv is None if self.cache_mode == "gru" else kv.shape[-2] == self.steps + 1
        if not valid_kv or self.cache.next_rnn_hiddens is None:
            raise RuntimeError('Temporal cache lost alignment with the physics episode')
        self.kv_bytes = 0 if kv is None else kv.numel() * kv.element_size()
        torch.cuda.synchronize(model.dev)
        self.seconds += time.perf_counter() - start
        return result.latents[:, -1].float(), result.proprio[:, -1].float()


class Models:
    def __init__(self, root, device):
        irasim = root.parent / 'IRASim'
        paths = {
            '--policy-bundle': irasim / 'results/action_expert/tokenworld_qwenoft_success_cam_high_teacher',
            '--codec-checkpoint': root / 'outputs/svae_d16_3views/best.pt',
            '--decoder-checkpoint': root / 'outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt',
            '--world-model-checkpoint': root / 'outputs/world_model_svae_d16_cam_high_100k/checkpoint_step_0100000.pt',
            '--adapter-checkpoint': root / 'outputs/deepstack_adapter_all50_20261031/adapter_best.pt',
            '--cache': irasim / 'cache/robotwin_all50_symmetric_qwen_svae_dim16_50hz',
        }
        path = paths.__getitem__
        self.dev=device
        self.policy,_,_=cl.load_policy(path('--policy-bundle'),device)
        self.stats=json.loads((path('--policy-bundle')/'model/dataset_statistics.json').read_text())
        self.codec,_=load_codec_checkpoint(path('--codec-checkpoint'),device=device)
        self.codec.eval()
        self.decoder=load_external_decoder(str(path('--decoder-checkpoint')),device=str(device),freeze=True)
        self.wm=WorldModelRollout(str(path('--world-model-checkpoint')),device=str(device),num_steps=16)
        self.adapter=DeepStackAdapter().to(device)
        self.adapter.load_state_dict(torch.load(path('--adapter-checkpoint'),map_location='cpu',weights_only=False)['model'])
        self.adapter.eval().requires_grad_(False)
        self.mean,self.std,self.action_mean,self.action_std=cl.load_latent_stats(path('--cache'),'observation.images.cam_high',device)
        self.assets={flag:str(path(flag)) for flag in ('--policy-bundle','--codec-checkpoint','--world-model-checkpoint','--adapter-checkpoint','--decoder-checkpoint','--cache')}

    def native(self,image,task):
        out=self.policy.predict_action(examples=[dict(image=[image],lang=task+'\nTrajectory outcome: success.')])
        return torch.as_tensor(out['normalized_actions'],device=self.dev,dtype=torch.float32)

    @torch.no_grad()
    def initial(self,image,qpos,task):
        image=image.resize((384,288),Image.Resampling.BILINEAR)
        vl=self.policy.qwen_vl_interface
        inp=vl.build_qwenvl_inputs(images=[[image]],instructions=[cl.action_instruction(self.policy,task,'success')])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            merged,_=vl.model.model.get_image_features(inp['pixel_values'].to(self.dev),inp['image_grid_thw'].to(self.dev))
            _,z,_=self.codec.encode_with_posterior(list(merged),sample_posterior=False)
        z=z.reshape(1,1,108,16).bfloat16().float()
        proprio=(torch.tensor(qpos,device=self.dev).float()-self.action_mean)/self.action_std
        return (z-self.mean)/self.std,proprio[None],self.native(image,task)

    @torch.no_grad()
    def query(self,state,method,template,task):
        raw=state.reshape(1,108,16)*self.std+self.mean
        if method=='adapter':
            return adapter_policy_actions(self.policy,self.codec,self.adapter,template,raw)
        # Match the codec's evaluation kernel used in the prior controlled rollout.
        with torch.enable_grad():
            merger=self.codec.decode(raw,[108])[0].reshape(1,108,2560).detach()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            rgb=self.decoder(merger)[0][0].float().cpu()
        pixels=((rgb+1)/2).clamp(0,1).permute(1,2,0).mul(255).round().byte().numpy()
        return self.native(Image.fromarray(pixels).resize((384,288),Image.Resampling.BILINEAR),task)

    @torch.inference_mode()
    def advance(self,state,proprio,context,seed,n):
        if len(context)<16:
            context=torch.cat([context,context[-1:].expand(16-len(context),-1)])
        d,c=cl.action_to_model_inputs(context,self.action_mean,self.action_std)
        torch.manual_seed(seed)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=self.wm.rollout(state,discrete_actions=d.unsqueeze(0),continuous_actions=c.unsqueeze(0),init_proprio=proprio)
        return out.latents[:,n].float(),out.proprio[:,n].float()


def read_msg(proc):
    data=proc.stdout.readline()
    if not data:
        raise RuntimeError(f'Simulator closed its output (exit {proc.wait()}); see physics log')
    return json.loads(data)


def branch(models,args,task,seed,method,folder,other_reset=None, *, advance=None, sim_env=None):
    if advance is None:
        # Each policy branch has its own recurrent state, scoped to one episode.
        mode = getattr(args, 'cache_mode', 'gru')
        advance = models.advance if mode == 'reset' else EpisodeStream(models, mode).advance
    cmd=['bash',str(HERE/'sim_python.sh'),str(HERE/'robotwin_physics_worker.py'),'--sim-root',str(args.sim_root),'--task',task,'--seed',str(seed)]
    with (folder/f'{method}.log').open('w') as log, subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=sim_env) as sim:
        try:
            reset=read_msg(sim)
            if reset.get('invalid_reset'):
                if other_reset is not None and reset != other_reset:
                    raise ValueError('Unstable reset differs between paired branches')
                if sim.wait(timeout=60):
                    raise RuntimeError('Unstable reset did not close cleanly')
                return reset, None
            img=base64.b64decode(reset.pop('image'))
            if abs(reset['timestep']*5-0.02)>1e-6:raise ValueError('Simulator and WM action clocks differ')
            if other_reset is not None and reset!=other_reset:
                (folder/'unmatched_reset.json').write_text(json.dumps(reset))
                raise ValueError('Paired physical reset, RGB or step budget differs')
            (folder/f'{method}.png').write_bytes(img)
            state,proprio,first=models.initial(Image.open(io.BytesIO(img)).convert('RGB'),reset['qpos'],task)
            template=cl.build_policy_template(models.policy,task,'success')
            step=0
            success=False
            actions=[]
            query_steps=[]
            while step<reset['limit'] and not success:
                pred=first if step==0 else models.query(state,method,template,task)
                q=cl.policy_to_cache_order(pred,models.stats)[0].detach()
                # q[0] is the current-time context. WM destinations and physical targets
                # are q1..q49, repeating q49 at the 50-step boundary of the 50-row chunk.
                q=torch.cat([q,q[-1:]],dim=0)
                assert q.shape==(51,14) and torch.isfinite(q).all()
                query_steps.append(step)
                print(f"{task} seed={seed} {method} policy query at step {step}",flush=True)
                offset=0
                while offset<50 and step<reset['limit'] and not success:
                    n=min(15,50-offset,reset['limit']-step)
                    context=q[offset:offset+n+1]
                    targets=context[1:].cpu().tolist()
                    sim.stdin.write(json.dumps(targets)+'\n');sim.stdin.flush()
                    msg=read_msg(sim)
                    used=msg['consumed']
                    assert 0<used<=n and msg['steps']==step+used
                    actions.extend(targets[:used])
                    success=msg['success']
                    if not success and step+used<reset['limit']:
                        assert used==n
                        state,proprio=advance(state,proprio,context,seed+step,n)
                    step+=used
                    offset+=used
            sim.stdin.close()
            if sim.wait(timeout=60): raise RuntimeError('Simulator did not close cleanly')
            np.save(folder/f'{method}.npy',np.asarray(actions,dtype=np.float32))
            return reset,dict(method=method,success=success,steps=step,query_steps=query_steps)
        finally:
            if sim.poll() is None:sim.kill()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--sim-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed-start',type=int,default=100000);p.add_argument('--seeds',type=int,default=1)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--cache-mode', choices=('gru', 'full', 'reset'), default='gru',
                   help='Temporal state retained across WM segments (default: GRU hidden only)')
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    a=p.parse_args()
    assert a.seeds>0 and 0<=a.rank<a.world_size
    torch.set_num_threads(2);dev=torch.device(a.device);torch.cuda.set_device(dev)
    models=Models(a.release,dev)
    for idx,(task,seed) in enumerate((t,s) for s in range(a.seed_start,a.seed_start+a.seeds) for t in a.tasks):
        if idx%a.world_size!=a.rank:continue
        folder=a.output/task/str(seed)
        folder.mkdir(parents=True,exist_ok=False)
        first,result_a=branch(models,a,task,seed,'adapter',folder)
        (folder/'reset.json').write_text(json.dumps(first))
        _,result_b=branch(models,a,task,seed,'rgb_bridge',folder,first)
        if first.get('invalid_reset'):
            (folder/'pair.json').write_text(json.dumps(dict(task=task,seed=seed,identical_reset=True,invalid_reset=True,cache_mode=a.cache_mode)))
            continue
        one=np.load(folder/'adapter.npy');two=np.load(folder/'rgb_bridge.npy')
        n=min(50,len(one),len(two))
        assert np.array_equal(one[:n],two[:n])
        record=dict(task=task,seed=seed,protocol='first_obs_only_shared_first_native_action_chunk',
                    cache_mode=a.cache_mode,shader='default',ctrl_hz=50,identical_reset=True,shared_first_actions=n,
                    assets=models.assets,adapter=result_a,rgb_bridge=result_b)
        (folder/'pair.json').write_text(json.dumps(record,indent=2))
        print(f"{task} {seed}: {result_a}, {result_b}",flush=True)

if __name__=='__main__':main()
