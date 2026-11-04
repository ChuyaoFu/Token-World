"""Check every requested task reset and native success predicate before GPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--tasks', nargs='+')
    a = p.parse_args()
    tasks = a.tasks or json.loads((HERE / 'tasks.json').read_text())
    folder = a.output / 'preflight'
    folder.mkdir(parents=True, exist_ok=True)

    def check(task):
        invalid = []
        for seed in range(a.seed, a.seed + 100):
            command = ['bash', str(ROOT / 'scripts/robotwin_sim/sim_python.sh'), str(HERE / 'physics_video.py'),
                '--task', task, '--seed', str(seed), '--sim-root', str(a.sim_root),
                '--frames', str(folder / (task + '.npz')), '--reset-only']
            log_path = folder / f'{task}_{seed}.log'
            with log_path.open('w') as log:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(f'{task}: reset preflight failed; see {log_path}')
            lines = result.stdout.splitlines()
            assert len(lines) == 1, (task, result.stdout[-1000:])
            row = json.loads(lines[0])
            row.pop('image', None)
            if row.get('invalid_reset'):
                invalid.append(seed)
                print('UNSTABLE RESET', task, seed, flush=True)
                continue
            assert row['limit'] > 0
            row['unstable_seeds'] = invalid
            print('RESET OK', task, seed, flush=True)
            return task, row
        raise RuntimeError(f'{task}: no stable reset in 100 preflight attempts')

    with ThreadPoolExecutor(4) as pool:
        rows = dict(pool.map(check, tasks))
    (folder / 'report.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
