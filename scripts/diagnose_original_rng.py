"""Replay selected episodes at their original full-evaluation RNG offsets.

For the one-GT-frame, 16-frame RoboTwin protocol. Skipped episodes consume only
the exact generation noise draws, whose equivalence is checked on every real
step. Use the regular open_loop arguments, including --max-episodes 100.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tokenworld.evaluation import open_loop as ev
from tokenworld.evaluation.rollout import WorldModelRollout

tasks=set(os.environ.get("TOKENWORLD_DIAG_TASKS",
    "put_object_cabinet,turn_switch,shake_bottle_horizontally").split(","))
original_select=ev.select_episode_indices
original_episode=ev._run_episode
original_step=WorldModelRollout.step
offsets, step_audit = {}, []

def noise_draws(latent, state):
    # generate(): context noise, new-frame noise, appended-context noise.
    for _ in range(3):
        torch.randn_like(latent)
        torch.randn_like(state)

def checked_step(self,x,d,c,state):
    before=torch.cuda.get_rng_state(x.device)
    result=original_step(self,x,d,c,state)
    after=torch.cuda.get_rng_state(x.device)
    torch.cuda.set_rng_state(before,x.device)
    noise_draws(x.unsqueeze(1),state.unsqueeze(1))
    expected=torch.cuda.get_rng_state(x.device)
    torch.cuda.set_rng_state(after,x.device)
    if not torch.equal(expected,after):
        raise RuntimeError("Noise-only replay does not match generation RNG consumption")
    step_audit.append({"before":hashlib.sha256(before.numpy().tobytes()).hexdigest(),
                       "after":hashlib.sha256(after.numpy().tobytes()).hexdigest()})
    return result

def selected(records,indices,maximum,selection,seed):
    full=original_select(records,indices,maximum,selection,seed)
    horizon=int(sys.argv[sys.argv.index("--max-steps")+1])
    assert horizon==16
    result=[]; skipped=0
    for i in full:
        if records[i]["task"] in tasks:
            offsets[i]=skipped;skipped=0;result.append(i)
        else:
            skipped+=min(horizon,int(records[i]["num_frames"])-1)
    return result

def episode(**kw):
    assert kw["initial_context_frames"]==1
    ds=kw["dataset"]; device=kw["device"]
    x=torch.empty((1,1,len(ds.view_indices),ds.tokens_per_view,ds.latent_channels),device=device)
    state=torch.empty((1,1,14),device=device)
    skipped=offsets[kw["record_index"]]
    for _ in range(skipped):
        noise_draws(x,state)
    begin=len(step_audit)
    result=original_episode(**kw)
    result[0]["original_rng_replay"]={"skipped_generation_steps":skipped,
        "noise_consumption_verified":True,"step_rng":step_audit[begin:]}
    return result

WorldModelRollout.step=checked_step
ev.select_episode_indices=selected
ev._run_episode=episode
if __name__=="__main__":
    ev.main()
