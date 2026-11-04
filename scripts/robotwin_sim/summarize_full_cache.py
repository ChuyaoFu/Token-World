"""Compare a recurrent policy and the archived reset-cache policy on a fixed panel."""
import argparse
import json
from pathlib import Path
from statistics import median


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-mode', choices=('full', 'gru'), default='full')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    a = p.parse_args()
    summary = {}
    for task in a.tasks:
        records = []
        invalid = []
        for seed in a.seeds:
            path = a.output / task / str(seed) / f'{a.cache_mode}_cache.json'
            record = json.loads(path.read_text())
            if (record['task'], record['seed'], record['method']) != (task, seed, f'wm_{a.cache_mode}_cache_adapter'):
                raise ValueError(f'Unexpected receipt: {path}')
            if record['invalid_reset']:
                invalid.append(seed)
            else:
                records.append(record)
        if not records:
            raise ValueError(f'No valid episodes for {task}')
        gains = [r['seed'] for r in records if r['result']['success'] and not r['old_success']]
        losses = [r['seed'] for r in records if not r['result']['success'] and r['old_success']]
        summary[task] = dict(cache_mode=a.cache_mode, valid=len(records), invalid=invalid,
                             reset_successes=sum(r['old_success'] for r in records),
                             gained_seeds=gains, lost_seeds=losses,
                             median_episode_seconds=median(r['elapsed_seconds'] for r in records),
                             wm_seconds_per_generated_step=sum(r['wm_seconds'] for r in records) /
                                 max(1,sum(r['wm_steps'] for r in records)),
                             max_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
                             max_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in records))
        summary[task][f'{a.cache_mode}_successes'] = sum(r['result']['success'] for r in records)
    text = json.dumps(summary, indent=2)
    (a.output/'summary.json').write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
