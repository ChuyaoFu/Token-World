"""Compare native true-RGB feedback with a finished paired RoboTwin/TokenWorld run."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import numpy as np
from PIL import Image
import torch
from tokenworld.evaluation import closed_loop as cl
from robotwin_wm_pair import TASKS, HERE, read_msg


def cabinet_replay_stage(source, directory):
    """Replay the recorded asset ID after the original RNG draw (glob order varies by mount)."""
    stage = directory / 'sim'
    stage.mkdir()
    shutil.copytree(source / 'envs', stage / 'envs')
    for name in ('assets', 'task_config', 'aloha_maniskill_sim'):
        (stage / name).symlink_to(source / name)
    path = stage / 'envs/put_object_cabinet.py'
    text = path.read_text()
    draw = '        self.selected_model_id = np.random.choice(available_model_ids)'
    assert text.count(draw) == 1
    path.write_text(text.replace(draw, draw + '''
        assert self.selected_modelname == os.environ["TW_NATIVE_ASSET_NAME"]
        replay_id = int(os.environ["TW_NATIVE_ASSET_ID"])
        assert replay_id in available_model_ids
        self.selected_model_id = replay_id'''))
    return stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sim-root', type=Path, required=True)
    p.add_argument('--policy-bundle', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    policy, _, _ = cl.load_policy(a.policy_bundle, dev)
    stats = json.loads((a.policy_bundle / 'model/dataset_statistics.json').read_text())
    lock = threading.Lock()
    work = []
    a.output.mkdir(parents=True, exist_ok=True)
    for seed in range(100000, 100025):
        for task in TASKS:
            pair = json.loads((a.panel/task/str(seed)/'pair.json').read_text())
            assert pair['task'] == task and pair['seed'] == seed and pair['identical_reset']
            if not pair.get('invalid_reset'):
                work.append(pair)
    assert len(work)==63

    def episode(pair, stage):
        task, seed = pair['task'], pair['seed']
        original = a.run/task/str(seed)
        expected = json.loads((original/'reset.json').read_text())
        folder = a.output/task/str(seed)
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder/'native.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            assert (result['task'], result['seed'], result['pair']) == (task, seed, [pair[m]['success'] for m in ('adapter','rgb_bridge')])
            return result
        ref = np.load(original/'adapter.npy')
        env = dict(os.environ)
        if task == 'put_object_cabinet':
            assets = expected['physics']['task_assets']
            env.update(TW_NATIVE_ASSET_ID=assets['selected_model_id'], TW_NATIVE_ASSET_NAME=assets['selected_modelname'])
        command = ['bash', str(HERE/'sim_python.sh'), str(HERE/'robotwin_physics_worker.py'), '--sim-root', str(stage),
                   '--task', task, '--seed', str(seed), '--live-rgb']
        with (folder/'physics.log').open('w') as log, subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                                       stderr=log, text=True, bufsize=1) as proc:
            try:
                reset = read_msg(proc)
                image = reset.pop('image', None)
                if reset != expected or image is None:
                    (folder/'unmatched.json').write_text(json.dumps(reset))
                    raise ValueError(f'{task}/{seed}: physical state, image, or step budget differs')
                steps, queries, actions = 0, [], []
                while True:
                    rgb = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
                    frame_hash = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
                    if steps==0: assert frame_hash==reset['rgb_sha256']
                    pil = rgb.resize((384,288),Image.Resampling.BILINEAR)
                    with lock, torch.inference_mode(), torch.cuda.device(dev):
                        prediction = policy.predict_action(examples=[dict(image=[pil],lang=task+'\nTrajectory outcome: success.')])
                        q = torch.as_tensor(prediction['normalized_actions'], dtype=torch.float32)
                        physical = cl.policy_to_cache_order(q,stats)[0].cpu().numpy()
                    assert physical.shape==(50,14) and np.isfinite(physical).all()
                    commands = np.concatenate([physical[1:], physical[-1:]])[:reset['limit']-steps]
                    if steps==0:
                        n = min(50,len(ref))
                        assert np.array_equal(commands[:n],ref[:n]), (task,seed,'first action mismatch')
                    queries.append(dict(step=steps,rgb_sha256=frame_hash))
                    proc.stdin.write(json.dumps(commands.tolist())+'\n')
                    proc.stdin.flush()
                    msg=read_msg(proc)
                    used=msg['consumed']
                    assert 0<used<=len(commands) and steps+used==msg['steps']
                    actions.extend(commands[:used])
                    steps+=used
                    if msg['success'] or steps==reset['limit']: break
                    assert used==50
                    image=msg['image']
                proc.stdin.close()
                if proc.wait(timeout=60): raise RuntimeError(f'physics exit for {task}/{seed}')
            finally:
                if proc.poll() is None: proc.kill()
        result=dict(task=task,seed=seed,success=msg['success'],steps=steps,identical_reset=True,matched_initial_actions=n,
                    queries=queries,pair=[pair[m]['success'] for m in ('adapter','rgb_bridge')])
        np.save(folder/'native.npy',np.asarray(actions))
        temp=folder/'native.tmp'
        temp.write_text(json.dumps(result,indent=2))
        temp.replace(result_path)
        print(task,seed,steps,msg['success'],flush=True)
        return result

    with tempfile.TemporaryDirectory(prefix='tw-native-rgb-') as tmp:
        stage=cabinet_replay_stage(a.sim_root.resolve(),Path(tmp))
        with ThreadPoolExecutor(a.workers) as pool:
            futures=[pool.submit(episode,pair,stage) for pair in work]
            results=[]
            errors=[]
            for f in as_completed(futures):
                try: results.append(f.result())
                except Exception as exc:
                    errors.append(str(exc)); print('FAILED',repr(exc),flush=True)
            if errors: raise RuntimeError(f'{len(errors)} incomplete: {errors[:4]}')
    assert len(results)==len(work)
    summary={}
    for task in TASKS:
        rows=[r for r in results if r['task']==task]
        summary[task]=dict(valid=len(rows),native=sum(r['success'] for r in rows))
        for i,m in enumerate(('adapter','rgb_bridge')):
            summary[task][m]=dict(success=sum(r['pair'][i] for r in rows),
                                  native_only=sum(r['success'] and not r['pair'][i] for r in rows),
                                  wm_only=sum(not r['success'] and r['pair'][i] for r in rows))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__': main()
