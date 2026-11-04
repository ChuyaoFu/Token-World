"""Evaluate the full-cache adapter policy against archived reset-cache RoboTwin episodes."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from evaluate_native_rgb import cabinet_replay_stage
from robotwin_wm_pair import EpisodeStream as PairedEpisodeStream, Models, TASKS, branch

# Archived full-cache diagnostics import this name and call it without a mode.
class EpisodeStream(PairedEpisodeStream):
    def __init__(self, models, cache_mode="full"):
        super().__init__(models, cache_mode)


def run(args):
    dev = torch.device(args.device)
    torch.set_num_threads(2)
    torch.cuda.set_device(dev)
    models = Models(args.release.resolve(), dev)
    reference = args.reference.resolve()
    panel = args.panel.resolve() if args.panel else reference
    cases = [(task, seed) for seed in args.seeds for task in args.tasks]
    with tempfile.TemporaryDirectory(prefix='tokenworld-full-cache-') as scratch:
        stage = (cabinet_replay_stage(args.sim_root.resolve(), Path(scratch))
                 if 'put_object_cabinet' in args.tasks else args.sim_root.resolve())
        branch_args = argparse.Namespace(sim_root=stage)
        for task, seed in cases[args.rank::args.world_size]:
            key = Path(task) / str(seed)
            folder = args.output / key
            folder.mkdir(parents=True, exist_ok=True)
            receipt = folder / f'{args.cache_mode}_cache.json'
            other_mode = 'full' if args.cache_mode == 'gru' else 'gru'
            if (folder / f'{other_mode}_cache.json').exists():
                raise ValueError(f'Choose a separate output directory for {args.cache_mode} mode')
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if record['reference'] != str(reference) or record['panel'] != str(panel):
                    raise ValueError(f'Conflicting references in {receipt}')
                continue
            source = reference / key
            pair = json.loads((panel / key / 'pair.json').read_text())
            if pair.get('cache_mode', 'reset') != 'reset':
                raise ValueError(f'Expected a reset-mode comparison panel for {key}')
            expected = json.loads((source / 'reset.json').read_text())
            if pair['task'] != task or pair['seed'] != seed or not pair['identical_reset']:
                raise ValueError(f'Unverified reference for {key}')
            record = dict(task=task, seed=seed, method=f'wm_{args.cache_mode}_cache_adapter', reference=str(reference),
                          panel=str(panel), invalid_reset=bool(pair.get('invalid_reset')))
            if not record['invalid_reset']:
                if pair['assets'] != models.assets:
                    raise ValueError(f'Model paths differ from reference for {key}')
                env = os.environ.copy()
                if task == 'put_object_cabinet':
                    asset = expected['physics']['task_assets']
                    env.update(TW_NATIVE_ASSET_NAME=asset['selected_modelname'], TW_NATIVE_ASSET_ID=asset['selected_model_id'])
                torch.cuda.reset_peak_memory_stats(dev)
                stream = EpisodeStream(models, args.cache_mode)
                start = time.perf_counter()
                reset, result = branch(models, branch_args, task, seed, 'adapter', folder, expected,
                                       advance=stream.advance, sim_env=env)
                actions = np.load(folder / 'adapter.npy')
                ref_actions = np.load(source / 'adapter.npy')
                n = min(50, len(actions), len(ref_actions))
                if n == 0 or not np.array_equal(actions[:n], ref_actions[:n]):
                    raise ValueError(f'Initial targets differ from reference for {key}')
                record.update(result=result, reset_rgb_sha256=reset['rgb_sha256'], shared_actions=n,
                              old_success=pair['adapter']['success'], old_steps=pair['adapter']['steps'],
                              elapsed_seconds=time.perf_counter()-start, wm_steps=stream.steps,
                              wm_seconds=stream.seconds, kv_bytes=stream.kv_bytes,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(dev),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(dev))
                del stream
            temp = receipt.with_suffix('.tmp')
            temp.write_text(json.dumps(record, indent=2))
            temp.replace(receipt)
            print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--panel', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sim-root', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--tasks', choices=TASKS, nargs='+', default=TASKS)
    p.add_argument('--cache-mode', choices=('full','gru'), default='full',
                   help='State retained between action segments')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    args = p.parse_args()
    if not 0 <= args.rank < args.world_size or len(set(args.seeds)) != len(args.seeds) or len(set(args.tasks)) != len(args.tasks):
        p.error('Expected unique seeds/tasks and a valid worker rank')
    run(args)


if __name__ == '__main__':
    main()
