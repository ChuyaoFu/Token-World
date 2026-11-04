"""Run the original and release native training entrypoints under identical controls.

The observer does not replace forward, backward, Adam, or EMA updates. Every
input, gradient, model tensor and optimizer tensor is SHA256-fingerprinted.
A digest mismatch fails the comparison; it is never accepted by tolerance.
Outputs go to a fresh directory; source trees and production checkpoints stay intact.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--processes', type=int, default=1)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resume-checkpoint', type=Path,
                   help='Exercise each native checkpoint loader and run this many additional steps')
    p.add_argument('--worker', choices=['original', 'release'])
    a = p.parse_args()
    if a.steps < 1 or a.processes < 1 or a.workers < 0:
        p.error('steps/processes must be positive and workers nonnegative')
    for k in ('original_root', 'config', 'output'):
        setattr(a, k, getattr(a, k).resolve())
    if a.resume_checkpoint:
        a.resume_checkpoint = a.resume_checkpoint.resolve()
    return a


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(a):
    import numpy as np
    import torch
    import yaml
    release_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(release_root))
    sys.path.insert(0, str(a.original_root))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    out = a.output / a.worker
    out.mkdir(exist_ok=True)
    trace_path = out / f'trace.rank{rank}.jsonl'
    trace = trace_path.open('w')

    def seed(value):
        random.seed(value)
        np.random.seed(value % 2**32)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def tensor_info(t):
        t = t.detach().contiguous().cpu()
        return {'shape': list(t.shape), 'dtype': str(t.dtype),
                'sha256': hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                'finite': bool(torch.isfinite(t).all())}

    def tree(obj):
        if torch.is_tensor(obj):
            return tensor_info(obj)
        if isinstance(obj, dict):
            return {str(k): tree(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [tree(v) for v in obj]
        return obj

    def rng():
        return {'python': hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                'numpy': hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                'torch_cpu': tensor_info(torch.get_rng_state())['sha256'],
                'torch_device': tensor_info(torch.cuda.get_rng_state())['sha256']}

    module_name = 'dreamer4.train_dreamer_vlm' if a.worker == 'original' else 'tokenworld.dynamics.train'
    entry = importlib.import_module(module_name)
    cls = entry.BehaviorCloneTrainer
    native_init = cls.__init__
    native_load = cls.load_checkpoint
    observation = {'step': 0, 'inside_forward': 0}
    model_module = importlib.import_module(entry.DynamicsWorldModel.__module__)
    native_sample_prob = model_module.sample_prob

    def sample_prob(prob):
        answer = native_sample_prob(prob)
        if observation['inside_forward']:
            observation['row'].setdefault('probability_draws', []).append({'p': prob, 'value': bool(answer)})
        return answer
    model_module.sample_prob = sample_prob

    def observed_init(self, model, *args, **kwargs):
        # Compare native initialization as well as explicitly loading the shared state.
        native_initial = tree(model.state_dict())
        initial_path = a.output / 'initial_model.pt'
        if a.worker == 'original' and rank == 0:
            torch.save(model.state_dict(), initial_path)
        if a.worker == 'release':
            model.load_state_dict(torch.load(initial_path, map_location='cpu', weights_only=True), strict=True)
        native_init(self, model, *args, **kwargs)
        unwrapped = self.accelerator.unwrap_model(self.model)
        initialized = tree(unwrapped.state_dict())
        sources = {}
        for module in (entry, model_module, importlib.import_module(cls.__module__),
                       importlib.import_module(type(self.dataset).__module__)):
            path = Path(inspect.getfile(module)).resolve()
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {'native_initial': native_initial, 'shared_initial': initialized,
                    'requires_grad': {n: p.requires_grad for n, p in unwrapped.named_parameters()},
                    'optimizer_initial': tree(self.optim.state_dict()),
                    'optimizer_class': type(getattr(self.optim, 'optimizer', self.optim)).__name__,
                    'max_grad_norm': self.max_grad_norm,
                    'precision': self.accelerator.mixed_precision,
                    'batch_size': self.batch_size, 'world_size': self.accelerator.num_processes,
                    'dataset_length': len(self.dataset),
                    'dataset_records': hashlib.sha256(json.dumps(self.dataset.records, sort_keys=True, default=str).encode()).hexdigest(),
                    'sources': sources, 'torch': torch.__version__,
                    'device': torch.cuda.get_device_name(), 'seed': a.seed}
        dump(out / f'initial.rank{rank}.json', metadata)
        # Both native DataLoaders receive the same RNG start, sampler and worker seeds.
        seed(a.seed + 100000)

        def pre_forward(mod, args_, kwargs_):
            if observation['inside_forward']:
                observation['inside_forward'] += 1
                return
            observation['inside_forward'] = 1
            step = observation['step']
            # Different deterministic noise per rank; original/release share each draw.
            noise_seed = a.seed + step + rank * 10000
            seed(noise_seed)
            observation['row'] = {'step': step, 'noise_seed': noise_seed,
                                  'batch': tree({'args': args_, 'kwargs': kwargs_}),
                                  'rng_before': rng()}
            observation['start'] = time.time()

        def post_forward(mod, args_, kwargs_, result):
            observation['inside_forward'] -= 1
            if observation['inside_forward']:
                return
            loss, parts, *_ = result
            row = observation['row']
            row['loss'] = float(loss.detach())
            row['loss_parts'] = {k: (float(v.detach()) if torch.is_tensor(v) else v)
                                 for k, v in zip(parts._fields, parts)}
            row['loss_tensors'] = tree({'loss': loss, 'parts': parts})
            row['rng_after_forward'] = rng()

        unwrapped.register_forward_pre_hook(pre_forward, with_kwargs=True)
        unwrapped.register_forward_hook(post_forward, with_kwargs=True)
        native_step = self.optim.step

        def observed_step(*args_, **kwargs_):
            row = observation['row']
            row['gradients'] = {n: (tensor_info(p.grad) if p.grad is not None else None)
                                for n, p in unwrapped.named_parameters()}
            result = native_step(*args_, **kwargs_)
            row['state_after_update'] = tree(unwrapped.state_dict())
            row['optimizer_after_update'] = tree(self.optim.state_dict())
            row['rng_after_update'] = rng()
            row['elapsed_seconds'] = time.time() - observation['start']
            trace.write(json.dumps(row, allow_nan=False) + '\n')
            trace.flush()
            print(f"CONTROL {a.worker} rank={rank} step={row['step']} loss={row['loss']:.9g} draws={row.get('probability_draws')} seconds={row['elapsed_seconds']:.1f}", flush=True)
            observation['step'] += 1
            return result
        self.optim.step = observed_step

    cls.__init__ = observed_init
    def observed_load(self, path):
        result = native_load(self, path)
        dump(out / f'resumed.rank{rank}.json', {
            'epoch': int(result[0]), 'global_step': int(result[1]),
            'best_val_loss': str(result[2]),
            'model': tree(self.accelerator.unwrap_model(self.model).state_dict()),
            'optimizer': tree(self.optim.state_dict()),
            'rng': rng(),
        })
        return result
    cls.load_checkpoint = observed_load
    config = yaml.safe_load(a.config.read_text())
    start_step, start_epoch = 0, 0
    if a.resume_checkpoint:
        payload = torch.load(a.resume_checkpoint, map_location='cpu', weights_only=False)
        start_step = int(payload.get('global_step', 0))
        start_epoch = int(payload.get('epoch', 0))
        del payload
    config['training'].update(num_epochs=1, max_train_steps=a.steps,
                              num_workers=a.workers, val_num_workers=0,
                              resume_from=None, checkpoint_every=None,
                              checkpoint_every_steps=None, checkpoint_folder=str(out))
    if a.resume_checkpoint:
        config['training'].update(num_epochs=start_epoch + 1,
                                  max_train_steps=start_step + a.steps,
                                  resume_from=str(a.resume_checkpoint))
    config['wandb']['enabled'] = False
    effective_path = out / f'controlled.rank{rank}.yaml'
    effective_path.write_text(yaml.safe_dump(config, sort_keys=False))
    sys.argv = [module_name, '--config', str(effective_path), '--checkpoint_folder', str(out)]
    seed(a.seed)
    entry.main()
    trace.close()
    if observation['step'] != a.steps:
        raise RuntimeError(f"Expected {a.steps} updates, got {observation['step']}")
    dump(out / f'complete.rank{rank}.json', {'status': 'complete', 'steps': observation['step']})


def compare(a):
    def read(side, filename):
        return json.loads((a.output / side / filename).read_text())
    checks = []
    def check(name, left, right):
        checks.append({'name': name, 'exact': left == right})
    for rank in range(a.processes):
        prefix = f'rank{rank}'
        old = read('original', f'initial.{prefix}.json')
        new = read('release', f'initial.{prefix}.json')
        for field in old:
            if field != 'sources':
                check(f'{prefix}.initial.{field}', old[field], new[field])
        for side, meta in [('original', old), ('release', new)]:
            check(f'{prefix}.{side}.world_size', meta['world_size'], a.processes)
            check(f'{prefix}.{side}.initial_state_preserved', meta['native_initial'], meta['shared_initial'])
            check(f'{prefix}.{side}.complete', read(side, f'complete.{prefix}.json')['steps'], a.steps)
        if a.resume_checkpoint:
            check(f'{prefix}.resumed_state', read('original', f'resumed.{prefix}.json'),
                  read('release', f'resumed.{prefix}.json'))
        rows = []
        for side in ('original', 'release'):
            rows.append([json.loads(x) for x in (a.output / side / f'trace.{prefix}.jsonl').read_text().splitlines()])
        check(f'{prefix}.trace_length', len(rows[0]), a.steps)
        check(f'{prefix}.matched_length', len(rows[0]), len(rows[1]))
        for old_row, new_row in zip(*rows):
            for key in old_row:
                if key != 'elapsed_seconds':
                    check(f"{prefix}.step{old_row['step']}.{key}", old_row[key], new_row[key])
            for side, row in [('original', old_row), ('release', new_row)]:
                # SHA equality must not conceal equal non-finite states.
                def finite(v):
                    if isinstance(v, dict):
                        return all((x is True if k == 'finite' else finite(x)) for k, x in v.items())
                    if isinstance(v, list):
                        return all(finite(x) for x in v)
                    return True
                check(f"{prefix}.step{row['step']}.{side}.finite", finite(row), True)
    if a.processes > 1:
        for side in ('original', 'release'):
            root_meta = read(side, 'initial.rank0.json')
            root_rows = [json.loads(x) for x in (a.output / side / 'trace.rank0.jsonl').read_text().splitlines()]
            for rank in range(1, a.processes):
                meta = read(side, f'initial.rank{rank}.json')
                check(f'{side}.rank{rank}.same_initial_as_rank0', root_meta['shared_initial'], meta['shared_initial'])
                rows = [json.loads(x) for x in (a.output / side / f'trace.rank{rank}.jsonl').read_text().splitlines()]
                for left, right in zip(root_rows, rows):
                    prefix = f"{side}.rank{rank}.step{left['step']}"
                    check(prefix + '.allreduced_gradients', left['gradients'], right['gradients'])
                    check(prefix + '.synchronized_parameters',
                          {n: left['state_after_update'][n] for n in root_meta['requires_grad']},
                          {n: right['state_after_update'][n] for n in root_meta['requires_grad']})
                    check(prefix + '.synchronized_optimizer', left['optimizer_after_update'], right['optimizer_after_update'])
                    # EMA buffers can differ between ranks after a forward pass;
                    # DDP broadcasts them at the next forward, so do not demand
                    # cross-rank buffer equality here.
    result = {'status': 'pass' if all(x['exact'] for x in checks) else 'fail',
              'steps': a.steps, 'processes': a.processes, 'seed': a.seed,
              'resume_checkpoint': str(a.resume_checkpoint) if a.resume_checkpoint else None,
              'comparison': 'Exact SHA256 of all tensor bytes, including gradients, buffers and Adam state; no numerical tolerance',
              'checks': checks, 'failed_checks': [x for x in checks if not x['exact']],
              'limits': 'Short controlled run in one runtime; does not reproduce the historical 100k trajectory or test eight-rank/multi-node topology.'}
    dump(a.output / 'comparison.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))
    return result['status'] == 'pass'


def main():
    a = parse_args()
    if a.worker:
        worker(a)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    dump(a.output / 'invocation.json', {'argv': sys.argv, 'environment': {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONHASHSEED']}})
    for side in ('original', 'release'):
        command = [sys.executable]
        if a.processes > 1:
            command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={a.processes}']
        command += [str(Path(__file__).resolve()), '--original-root', str(a.original_root),
                    '--config', str(a.config), '--output', str(a.output), '--steps', str(a.steps),
                    '--seed', str(a.seed), '--processes', str(a.processes), '--workers', str(a.workers), '--worker', side]
        if a.resume_checkpoint:
            command += ['--resume-checkpoint', str(a.resume_checkpoint)]
        env = dict(os.environ, PYTHONHASHSEED=str(a.seed), OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'), MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS', '4'))
        print('Starting', side, flush=True)
        with (a.output / f'{side}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not compare(a):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
