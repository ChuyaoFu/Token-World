"""First-observation-only TokenWorld closed loop versus identical-action simulation."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/robotwin_sim')]
from robotwin_wm_pair import Models, read_msg
from tokenworld.evaluation import closed_loop as cl


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def contract(a):
    sources = [HERE / n for n in ('run.py', 'physics_video.py', 'tasks.json')]
    sources += [ROOT / 'scripts/robotwin_sim/robotwin_wm_pair.py',
                ROOT / 'tokenworld/evaluation/closed_loop.py', ROOT / 'tokenworld/evaluation/rollout.py']
    checkpoints = [ROOT / p for p in ('outputs/svae_d16_3views/best.pt',
        'outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt',
        'outputs/world_model_svae_d16_cam_high_100k/checkpoint_step_0100000.pt',
        'outputs/deepstack_adapter_all50_20261031/adapter_best.pt')]
    policy = ROOT.parent / 'IRASim/results/action_expert/tokenworld_qwenoft_success_cam_high_teacher'
    cache = ROOT.parent / 'IRASim/cache/robotwin_all50_symmetric_qwen_svae_dim16_50hz'
    # Record policy payload identities without loading the multi-GB expert on CPU.
    policy_files = sorted(p for p in policy.rglob('*') if p.is_file() and p.suffix in
                          ('.json', '.yaml', '.yml', '.pt', '.bin', '.safetensors'))
    result = dict(protocol='same_initial_state_same_physical_actions_v1', tasks=a.tasks,
        per_task=a.per_task, seed_start=a.seed_start, max_attempts=a.max_attempts,
        max_steps=a.max_steps, policy_interface='native_pre_norm_deepstack_adapter',
        ctrl_hz=50, video_hz=25, query_every=50, execution_horizon=15, denoise_steps=16,
        cache_mode='gru', first_action='native RGB policy',
        action_alignment='context q0; targets q1..q49, then repeat q49',
        stopping='simulation success or native task limit; no simulated observations after frame zero',
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        checkpoint_sha256={str(p): digest(p) for p in checkpoints},
        cache_sha256={str(cache / n): digest(cache / n) for n in ('cache_spec.json', 'train_stats.pt')},
        policy_files={str(p.relative_to(policy)): [p.stat().st_size, p.stat().st_mtime_ns] for p in policy_files},
        simulator=json.loads((a.sim_root / 'ready.json').read_text()),
        task_code_sha256={t: digest(a.sim_root / 'envs' / (t + '.py')) for t in a.tasks})
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def render(folder, task, seed):
    with np.load(folder / 'simulation_rgb.npz') as s, np.load(folder / 'tokenworld_rgb.npz') as w:
        assert np.array_equal(s['frame_ids'], w['frame_ids'])
        assert s['rgb'].shape == w['rgb'].shape
        sim, wm, ids = s['rgb'], w['rgb'], s['frame_ids']
        assert np.array_equal(sim[0], wm[0])
        frames = []
        for i, (left, right) in enumerate(zip(wm, sim)):
            image = Image.new('RGB', (448, 268), 'white')
            image.paste(Image.fromarray(left), (0, 44))
            image.paste(Image.fromarray(right), (224, 44))
            draw = ImageDraw.Draw(image)
            draw.text((5, 3), f'{task} | seed {seed} | {ids[i] / 50:.2f}s', fill='black')
            draw.text((5, 23), 'TokenWorld closed loop', fill='black')
            draw.text((229, 23), 'Simulation / same actions', fill='black')
            frames.append(np.asarray(image))
        frames = np.stack(frames)
        for name, images in [('tokenworld.mp4', wm), ('simulation.mp4', sim), ('comparison.mp4', frames)]:
            dst = folder / name
            tmp = folder / (dst.stem + '.tmp.mp4')
            cl.write_video(images, tmp, 25)
            tmp.replace(dst)
        rows = [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, 7).round().astype(int))]
        Image.fromarray(np.concatenate(rows, axis=1)).save(folder / 'thumbnail.png')


class Stream:
    def __init__(self, models):
        self.m, self.cache = models, None

    @torch.inference_mode()
    def advance(self, state, proprio, context, seed):
        m = self.m
        d, c = cl.action_to_model_inputs(context, m.action_mean, m.action_std)
        torch.manual_seed(seed)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = m.wm.rollout(state, d[None], c[None], proprio, time_cache=self.cache,
                               return_time_cache=True, gru_only_boundary=True)
        self.cache = out.time_cache
        assert self.cache.next_kv_cache is None and self.cache.next_rnn_hiddens is not None
        return out.latents[:, 1:].float(), out.proprio[:, -1].float()


@torch.no_grad()
def episode(m, a, task, seed, folder, fingerprint):
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / 'result.json'
    if complete.exists():
        result = json.loads(complete.read_text())
        assert result['contract_sha256'] == fingerprint
        if result.get('invalid_reset'):
            return result
        for name in ('actions.npy', 'wm_contexts.npz', 'latents.npz', 'simulation_rgb.npz', 'tokenworld_rgb.npz',
                     'comparison.mp4', 'simulation.mp4', 'tokenworld.mp4', 'thumbnail.png'):
            assert (folder / name).stat().st_size > 0, (folder, name)
        print('REUSE', task, seed, flush=True)
        return result
    # Rendering can be resumed independently of policy, WM and physics.
    generated = folder / 'generation.json'
    if generated.exists():
        result = json.loads(generated.read_text())
        assert result['contract_sha256'] == fingerprint
        render(folder, task, seed)
        save_json(complete, result)
        return result
    torch.manual_seed(seed)
    np.random.seed(seed)
    command = ['bash', str(ROOT / 'scripts/robotwin_sim/sim_python.sh'), str(HERE / 'physics_video.py'),
        '--task', task, '--seed', str(seed), '--sim-root', str(a.sim_root),
        '--frames', str(folder / 'simulation_rgb.npz'), '--max-steps', str(a.max_steps)]
    with (folder / 'physics.log').open('w') as log, subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1) as sim:
        try:
            reset = read_msg(sim)
            result = dict(task=task, seed=seed, contract_sha256=fingerprint)
            if reset.get('invalid_reset'):
                if sim.wait(timeout=60):
                    raise RuntimeError('Unstable reset did not exit cleanly')
                result['invalid_reset'] = True
                save_json(complete, result)
                return result
            image = Image.open(io.BytesIO(base64.b64decode(reset.pop('image')))).convert('RGB')
            image.save(folder / 'initial.png')
            save_json(folder / 'reset.json', reset)
            assert abs(reset['timestep'] * 5 - .02) < 1e-6
            state, proprio, first = m.initial(image, reset['qpos'], task)
            stream = Stream(m)
            template = cl.build_policy_template(m.policy, task, 'success')
            latents, actions, queries = [state.cpu()], [], []
            contexts, kept, starts = [], [], []
            step, success = 0, False
            while step < reset['limit'] and not success:
                prediction = first if step == 0 else m.query(state, 'adapter', template, task)
                q = cl.policy_to_cache_order(prediction, m.stats)[0].detach()
                q = torch.cat([q, q[-1:]], dim=0)
                assert q.shape == (51, 14) and torch.isfinite(q).all()
                queries.append(step)
                print('QUERY', task, seed, 'step', step, flush=True)
                offset = 0
                while offset < 50 and step < reset['limit'] and not success:
                    n = min(15, 50 - offset, reset['limit'] - step)
                    context = q[offset:offset + n + 1]
                    targets = context[1:].cpu().numpy()
                    sim.stdin.write(json.dumps(targets.tolist()) + '\n')
                    sim.stdin.flush()
                    msg = read_msg(sim)
                    used = int(msg['consumed'])
                    assert 0 < used <= n and msg['steps'] == step + used
                    assert used == n or msg['success'] or msg['steps'] == reset['limit']
                    # Predict the full planned block independently of simulator
                    # termination. A terminal partial block retains only its prefix.
                    predicted, proprio = stream.advance(state, proprio, context, seed + step)
                    assert predicted.shape[1] == n
                    contexts.append(context.cpu().numpy())
                    kept.append(used)
                    starts.append(step)
                    predicted = predicted[:, :used]
                    latents.append(predicted.cpu())
                    state = predicted[:, -1:].float()
                    actions.extend(targets[:used])
                    step += used
                    offset += used
                    success = bool(msg['success'])
            sim.stdin.close()
            if sim.wait(timeout=60):
                raise RuntimeError('Simulator failed while writing frames')
        finally:
            if sim.poll() is None:
                sim.kill()
                sim.wait()
    z = torch.cat(latents, dim=1).to(m.dev)
    assert z.shape[1] == step + 1
    frame_ids = np.arange(0, step + 1, 2, dtype=np.int64)
    rgb = [np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))]
    for begin in range(1, len(frame_ids), 8):
        raw = z[0, frame_ids[begin:begin + 8]].reshape(-1, 108, 16) * m.std + m.mean
        with torch.enable_grad():
            merger = m.codec.decode(raw, [108] * len(raw))
            merger = torch.stack(merger).detach()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            decoded = m.decoder(merger)[0].float()
        rgb.extend(((decoded + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).mul(255).round().byte().cpu().numpy())
    np.save(folder / 'actions.npy', np.asarray(actions, dtype=np.float32))
    np.savez_compressed(folder / 'wm_contexts.npz', contexts=np.concatenate(contexts),
                        offsets=np.cumsum([0] + [len(c) for c in contexts]),
                        consumed=np.asarray(kept), start_steps=np.asarray(starts))
    np.savez_compressed(folder / 'latents.npz', normalized_latents=z[0].cpu().numpy(),
                        frame_ids=np.arange(step + 1))
    np.savez_compressed(folder / 'tokenworld_rgb.npz', rgb=np.stack(rgb), frame_ids=frame_ids)
    result.update(steps=step, success=success, query_steps=queries, frames_25hz=len(frame_ids),
                  actions_sha256=digest(folder / 'actions.npy'), identical_actions=True,
                  initial_rgb_sha256=reset['rgb_sha256'], native_limit=reset['native_limit'])
    save_json(generated, result)
    render(folder, task, seed)
    save_json(complete, result)
    print('DONE', task, seed, step, success, flush=True)
    return result


def summarize(a, fingerprint):
    tasks = {}
    for task in a.tasks:
        rows = []
        for seed in range(a.seed_start, a.seed_start + a.max_attempts):
            path = a.output / task / str(seed) / 'result.json'
            if not path.exists():
                break
            row = json.loads(path.read_text())
            assert row['contract_sha256'] == fingerprint
            rows.append(row)
            if sum(not r.get('invalid_reset') for r in rows) == a.per_task:
                break
        valid = [r for r in rows if not r.get('invalid_reset')]
        assert len(valid) == a.per_task, (task, len(valid), a.per_task)
        assert all((a.output / task / str(r['seed']) / 'comparison.mp4').stat().st_size for r in valid)
        tasks[task] = dict(videos=len(valid), seeds=[r['seed'] for r in valid],
                           invalid_seeds=[r['seed'] for r in rows if r.get('invalid_reset')],
                           simulation_successes=sum(r['success'] for r in valid))
    save_json(a.output / 'summary.json', dict(contract_sha256=fingerprint, tasks=tasks,
        total_videos=sum(t['videos'] for t in tasks.values()),
        interpretation='WM state closed-loop actions physically replayed, no simulated RGB feedback after initialization'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--per-task', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=100000)
    p.add_argument('--max-attempts', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    all_tasks = json.loads((HERE / 'tasks.json').read_text())
    a.tasks = a.tasks or all_tasks
    assert len(set(a.tasks)) == len(a.tasks) and set(a.tasks) <= set(all_tasks)
    assert 0 <= a.rank < a.world_size and 0 < a.per_task <= a.max_attempts and a.max_steps >= 0
    a.output, a.sim_root = a.output.resolve(), a.sim_root.resolve()
    path = a.output / 'contract.json'
    if a.prepare:
        new = contract(a)
        if path.exists():
            assert json.loads(path.read_text()) == new, 'Output belongs to a different experiment'
        else:
            save_json(path, new)
        print('Prepared', len(a.tasks), 'tasks x', a.per_task, flush=True)
        return
    saved = json.loads(path.read_text())
    for key in ('tasks', 'per_task', 'seed_start', 'max_attempts', 'max_steps'):
        assert saved[key] == getattr(a, key), key
    if a.summarize:
        summarize(a, saved['sha256'])
        return
    torch.set_num_threads(2)
    device = torch.device(f'cuda:{a.rank}')
    torch.cuda.set_device(device)
    m = Models(ROOT, device)
    for task in a.tasks[a.rank::a.world_size]:
        valid = 0
        for seed in range(a.seed_start, a.seed_start + a.max_attempts):
            row = episode(m, a, task, seed, a.output / task / str(seed), saved['sha256'])
            valid += not row.get('invalid_reset', False)
            if valid == a.per_task:
                break
        if valid != a.per_task:
            raise RuntimeError(f'{task}: only {valid}/{a.per_task} valid resets after {a.max_attempts} attempts')


if __name__ == '__main__':
    main()
