"""Paired action-fidelity diagnosis with step-level common random numbers.

Runs both checkpoints on one device, reusing the same teacher, codec, and policy.
Teacher-forced and GT-proprio modes are diagnostic controls, not open-loop scores.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tokenworld.evaluation import open_loop as ev
from tokenworld.evaluation.rollout import WorldModelRollout
from tokenworld.data.dataset import RobotwinSVAETokenDataset
from tokenworld.codec.common import load_codec_checkpoint, normalize_qwen_image_features


def metrics(ref, pred):
    a = ev.FidelityAccumulator()
    a.update(ref, pred)
    return a.report()


class PairedNoiseModel:
    def __init__(self, model, dataset, record_index, seed, mode):
        self.model, self.seed, self.mode = model, seed, mode
        self.calls, self.noise_states = 0, []
        p = dataset._load_episode(record_index)
        x = p["latents"][dataset.latent_method].float()[:, dataset.view_indices]
        self.latents = (x - dataset.latent_mean) / dataset.latent_std
        self.states = (p["states"].float() - dataset.action_mean) / dataset.action_std

    def step(self, x, d, c, state):
        index = self.calls
        self.calls += 1
        seed = self.seed + index
        random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
        if self.mode == "teacher_forced":
            x = self.latents[index:index+1].to(x.device)
        if self.mode in ("teacher_forced", "gt_proprio"):
            state = self.states[index:index+1].to(x.device)
        def digest():
            return hashlib.sha256(torch.cuda.get_rng_state(x.device).numpy().tobytes()).hexdigest()
        before = digest()
        y = self.model.step(x, d, c, state)
        self.noise_states.append({"seed": seed, "before": before, "after": digest()})
        return y


class TraceExpert:
    def __init__(self, expert):
        self.expert = expert
        self.actions_log, self.features_log = [], []

    def actions(self, task, outcome, state, features):
        value = self.expert.actions(task, outcome, state, features)
        self.actions_log.append(value.detach().cpu())
        self.features_log.append(features.detach().cpu())
        return value


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("cache", "baseline", "candidate", "codec-checkpoint", "qwen-model",
                 "raw-data-root", "policy-bundle", "selection", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--tasks", default="put_object_cabinet,turn_switch,shake_bottle_horizontally")
    p.add_argument("--seeds", default="3047,3048,3049")
    p.add_argument("--modes", default="autoregressive,teacher_forced,gt_proprio")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps", type=int, default=16)
    args = p.parse_args()
    modes = args.modes.split(",")
    if not set(modes) <= {"autoregressive", "teacher_forced", "gt_proprio"}:
        p.error("Unknown diagnostic mode")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "invocation.json").write_text(json.dumps({"command": ["bash", "scripts/run_python.sh", *sys.argv]}, indent=2) + "\n")
    ds = RobotwinSVAETokenDataset(args.cache, raw_data_path=args.raw_data_root,
        split="val", num_hist=1, num_pred=1, latent_method="symmetric_qwen", views=[0],
        normalize_latents=True, latent_stats_scope="per_view", normalize_action=True,
        verify_action_equals_state=True, sequence_interval=1, window_stride=1, tail_align=False)
    wanted = {(r["dataset"], int(r["episode_index"])) for r in
              json.loads(args.selection.read_text())["episodes"] if r["task"] in args.tasks.split(",")}
    indices = [i for i,r in enumerate(ds.records) if (r["dataset"],int(r["episode_index"])) in wanted]
    assert len(indices) == len(wanted) and len(indices) > 0
    codec, _ = load_codec_checkpoint(args.codec_checkpoint, device=device)
    codec.eval()
    teacher = ev._load_qwen_teacher(args.qwen_model, device)
    expert = ev.ActionExpert(args.policy_bundle, "robotwin", device)
    models = {label: WorldModelRollout(str(path), device=str(device), num_steps=16)
              for label,path in (("baseline",args.baseline),("candidate",args.candidate))}
    results, controls = [], []
    seeds = [int(x) for x in args.seeds.split(",")]
    original_extract = ev._extract_teacher_qwen
    stream = (args.output/"runs.jsonl").open("w")
    for i in indices:
        record = ds.records[i]
        episode_id = f"{record['dataset']}_{int(record['episode_index']):06d}"
        payload = ds._load_episode(i)
        frames = ds._load_raw_images(i, record, torch.arange(1,args.max_steps+1))
        refs = [original_extract(teacher, frame, normalize_qwen_image_features) for frame in frames]
        controls_features, controls_actions = ev.FidelityAccumulator(), ev.FidelityAccumulator()
        repeat_max, dimension_sse, dimension_energy = 0., torch.zeros(14,dtype=torch.float64), torch.zeros(14,dtype=torch.float64)
        gt_actions, decoded_features = [], []
        for k, ref in enumerate(refs,1):
            native = payload["latents"][ds.latent_method][k:k+1,ds.view_indices].float().to(device)
            decoded = ev._decode_qwen(codec,native)
            state=payload["states"][k].float()
            action=expert.actions(record["task"],record["outcome"],state,ref)
            repeated=expert.actions(record["task"],record["outcome"],state,ref)
            repeat_max=max(repeat_max,float((action-repeated).abs().max()))
            reconstructed=expert.actions(record["task"],record["outcome"],state,decoded)
            controls_features.update(ref,decoded); controls_actions.update(action,reconstructed)
            gt_actions.append(action.cpu()); decoded_features.append(decoded.cpu())
        control={"episode_id":episode_id,"task":record["task"],"outcome":record["outcome"],
                 "identical_feature_action_max_abs":repeat_max,
                 "codec_gt_feature_metrics":controls_features.report(),
                 "codec_gt_action_metrics":controls_actions.report()}
        controls.append(control)
        print("CONTROL",episode_id,json.dumps(control),flush=True)
        reference_cursor=[0]
        def cached_extract(*unused):
            index=reference_cursor[0];reference_cursor[0]+=1
            return refs[index]
        ev._extract_teacher_qwen=cached_extract
        saved={}
        for mode in modes:
            for seed in (seeds if mode=="autoregressive" else seeds[:1]):
                step_seed=int.from_bytes(hashlib.sha256(f"{episode_id}/{seed}".encode()).digest()[:4],"big")
                for label, model in models.items():
                    reference_cursor[0]=0
                    traced=TraceExpert(expert)
                    paired=PairedNoiseModel(model,ds,i,step_seed,mode)
                    result,_,_,ref_action,pred_action=ev._run_episode(
                        dataset=ds,record_index=i,world_model=paired,codec=codec,teacher=teacher,
                        action_expert=traced,normalize_features=normalize_qwen_image_features,
                        device=device,max_steps=args.max_steps,query_every=1,reference_mode="raw_teacher")
                    reference=torch.stack(traced.actions_log[1::2]).double()
                    predicted=torch.stack(traced.actions_log[::2]).double()
                    squared=(predicted-reference).square()
                    independent_nmse=float(squared.sum()/reference.square().sum())
                    assert abs(independent_nmse-result["action_metrics"]["nmse"])<1e-10
                    assert torch.equal(torch.cat(traced.actions_log[1::2]),torch.cat(gt_actions))
                    row={"checkpoint":label,"mode":mode,"seed":seed,"episode_id":episode_id,
                         "task":record["task"],"outcome":record["outcome"],
                         "feature":result["qwen_feature_metrics"],"action":result["action_metrics"],
                         "per_dimension_mse":squared.reshape(-1,14).mean(0).tolist(),
                         "per_dimension_nmse":(squared.reshape(-1,14).sum(0)/reference.square().reshape(-1,14).sum(0).clamp_min(1e-12)).tolist(),
                         "per_frame_action_nmse":[metrics(a,b)["nmse"] for a,b in zip(reference,predicted)],
                         "first_chunk_action":metrics(reference[...,0,:],predicted[...,0,:]),
                         "noise_states":paired.noise_states}
                    results.append(row);stream.write(json.dumps(row)+"\n");stream.flush()
                    print(label,mode,seed,episode_id,"action_nmse",row["action"]["nmse"],flush=True)
                    if mode=="autoregressive" and seed==seeds[0]:
                        saved[label]={"predicted_features":torch.stack(traced.features_log[::2]),
                                      "predicted_actions":predicted.float(),
                                      "reference_actions":reference.float()}
        torch.save({"controls":control,"raw_features":torch.stack([r.cpu() for r in refs]),
                    "codec_gt_features":torch.stack(decoded_features),**saved},args.output/(episode_id+".pt"))
        (args.output/"controls.json").write_text(json.dumps(controls,indent=2)+"\n")
        ev._extract_teacher_qwen=original_extract
    stream.close()
    for a,b in zip(results[::2],results[1::2]):
        assert a["noise_states"]==b["noise_states"],"Paired model RNG streams differ"
    (args.output/"diagnosis.json").write_text(json.dumps({
        "status":"complete","arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "protocol":"paired_step_noise_same_device_v1","same_noise_states_verified":True,
        "controls":controls,"runs":results},indent=2)+"\n")
    print("COMPLETE",args.output,flush=True)


if __name__=="__main__":
    main()
