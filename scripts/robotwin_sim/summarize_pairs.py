"""Aggregate a complete paired panel; unmatched or missing trials are evaluation errors."""
import argparse
import json
from pathlib import Path

TASKS = ('put_object_cabinet', 'shake_bottle_horizontally', 'turn_switch')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed-start', type=int, required=True)
    p.add_argument('--seeds', type=int, required=True)
    a = p.parse_args()
    summary = {}
    run_mode = None
    for task in TASKS:
        completed = []
        invalid = []
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            path = a.output / task / str(seed) / 'pair.json'
            result = json.loads(path.read_text())
            assert (result['task'], result['seed']) == (task, seed) and result['identical_reset']
            mode = result.get('cache_mode', 'reset')
            if run_mode is None:
                run_mode = mode
            if mode != run_mode:
                raise ValueError(f'Mixed cache modes in {a.output}: {run_mode} and {mode}')
            if result.get('invalid_reset'):
                invalid.append(seed)
                continue
            completed.append((result['adapter']['success'], result['rgb_bridge']['success']))
        if not completed:
            raise ValueError(f'No valid paired trials for {task}')
        n = len(completed)
        na = sum(x for x, y in completed)
        nb = sum(y for x, y in completed)
        summary[task] = dict(cache_mode=run_mode, valid_pairs=n, invalid_reset_seeds=invalid, requested=a.seeds,
                             adapter_successes=na, rgb_bridge_successes=nb,
                             adapter_rate=na/n, rgb_bridge_rate=nb/n, paired_difference=(na-nb)/n,
                             adapter_only=sum(x and not y for x,y in completed),
                             rgb_bridge_only=sum(y and not x for x,y in completed),
                             both_succeed=sum(x and y for x,y in completed))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__ == '__main__':
    main()
