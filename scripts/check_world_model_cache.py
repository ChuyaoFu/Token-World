"""Check finalized RoboTwin cache coverage and sample loading before training.

Run with scripts/run_python.sh. This checks file presence and sampled tensors,
not every tensor in the cache, and never changes cache contents.
"""
from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path

import torch
import yaml
from tokenworld.data.dataset import load_robotwin_svae_train_val


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=root / 'configs/robotwin/dynamics/svae_d16.yaml')
    parser.add_argument('--manifest', type=Path, default=root / 'configs/robotwin/splits/seed3047.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cache = args.cache.resolve()
    for name in ['cache_spec.json', 'index.json', 'train_stats.pt']:
        if not (cache / name).is_file():
            raise ValueError(f'Cache is not finalized: missing {cache / name}')
    spec = json.loads((cache / 'cache_spec.json').read_text())
    index = json.loads((cache / 'index.json').read_text())
    manifest = json.loads(args.manifest.read_text())
    cfg = yaml.safe_load(args.config.read_text())
    records = index['records']
    expected = {(dataset, int(ep), split, task)
                for dataset, data in manifest['datasets'].items()
                for task, task_data in data['tasks'].items()
                for split in ['train', 'val'] for ep in task_data[split]}
    actual = {(r['dataset'], int(r['episode_index']), r['split'], r['task']) for r in records}
    if actual != expected or len(actual) != len(records):
        raise ValueError('Cache episode assignments do not exactly match the manifest, or contain duplicates')
    missing = [r['path'] for r in records
               if not (cache / r['path']).is_file() or (cache / r['path']).stat().st_size == 0]
    if missing:
        raise ValueError(f'{len(missing)} cache files missing/empty, examples: {missing[:5]}')
    if spec.get('export_frame_stride') != 1 or spec.get('export_frame_rate_hz') != 50:
        raise ValueError('The reproduction recipe requires a 50-Hz cache with export_frame_stride=1')
    parameters = inspect.signature(load_robotwin_svae_train_val).parameters
    kwargs = {k: v for k, v in cfg['data'].items() if k in parameters and k != 'cache_dir'}
    datasets, _ = load_robotwin_svae_train_val(cache_dir=cache, **kwargs)
    expected_shape = (cfg['data']['num_hist'] + cfg['data']['num_pred'],
                      cfg['model']['num_video_views'], cfg['model']['num_latent_tokens'],
                      cfg['model']['dim_latent'])
    samples = {}
    for split, dataset in datasets.items():
        if not len(dataset):
            raise ValueError(f'{split} dataset has no usable windows')
        checked = sorted({0, len(dataset) // 2, len(dataset) - 1})
        for i in checked:
            sample = dataset[i]
            if tuple(sample['vlm_latents'].shape) != expected_shape:
                raise ValueError(f'{split} window {i} latent shape does not match the model: {sample["vlm_latents"].shape}')
            for key, value in sample.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all().item():
                    raise ValueError(f'{split} window {i}: {key} contains NaN or Inf')
        samples[split] = {'windows': len(dataset), 'checked_indices': checked,
                          'latent_shape': list(expected_shape)}
    print(json.dumps({'cache': str(cache), 'records': len(records),
                      'splits': dict(Counter(r['split'] for r in records)),
                      'all_index_files_present_and_nonempty': True,
                      'manifest_assignments_match': True,
                      'cache_codec_sha256': spec.get('codec_checkpoint_sha256'),
                      'samples': samples, 'status': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
