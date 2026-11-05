#!/usr/bin/env python3
"""Run two recorded-action WM segments using the bundled RoboTwin sample."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tokenworld.codec.common import load_codec_checkpoint
from tokenworld.codec.deepstack_adapter import DeepStackAdapter
from tokenworld.decoder.checkpoint import load_external_decoder
from tokenworld.evaluation.rollout import WorldModelRollout

ROOT = Path(__file__).resolve().parents[1]


def pixels(tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError('Non-finite model output')
    return tensor.float().add(1).mul(127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='PPU device exposed through the SDK CUDA API')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/minimal_inference')
    parser.add_argument('--sample', type=Path, default=ROOT / 'examples/robotwin/sample.npz')
    parser.add_argument('--seed', type=int, default=3047)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('This validated recipe requires a PPU SDK device; use scripts/run_python.sh.')
    # The PPU SDK CPU math path can crash in RoPE initialization with multiple
    # host threads. This small single-device demo does not need CPU parallelism.
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    start = time.monotonic()
    metadata = json.loads(args.sample.with_suffix('.json').read_text())
    if hashlib.sha256(args.sample.read_bytes()).hexdigest() != metadata['files'][args.sample.name]:
        raise RuntimeError('Example sample checksum mismatch')
    with np.load(args.sample, allow_pickle=False) as source:
        sample = {k: torch.from_numpy(source[k].copy()).to(args.device) for k in source.files}
    assert sample['initial_latent'].shape == (1, 1, 108, 16)
    assert sample['actions'].shape == (31, 14)
    assert all(torch.isfinite(v).all() for v in sample.values())
    mean, std = sample['latent_mean'], sample['latent_std']
    am, ast = sample['action_mean'], sample['action_std']
    assert (std > 0).all() and (ast > 0).all()
    manifest = json.loads((ROOT / 'configs/checkpoints.json').read_text())
    weights = {e['component']: ROOT / e['path'] for e in manifest['files']}
    for path in weights.values():
        if not path.is_file():
            parser.error(f'Missing {path}. Run scripts/download_checkpoints.py first.')
    print('Loading four published checkpoints', flush=True)
    wm = WorldModelRollout(str(weights['world_model']), device=args.device, num_steps=16)
    codec, _ = load_codec_checkpoint(weights['svae'], device=torch.device(args.device))
    codec.eval().requires_grad_(False)
    decoder = load_external_decoder(str(weights['rgb_decoder']), args.device, freeze=True)
    adapter = DeepStackAdapter().to(args.device).eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(weights['adapter'], map_location='cpu', weights_only=False)['model'])
    z = ((sample['initial_latent'] - mean) / std)[None]
    prop = ((sample['initial_proprio'] - am) / ast)[None]
    history = None
    all_latents, all_rgb, trace = [], [], []
    ci = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    for chunk, boundary in enumerate((0, 15)):
        act = sample['actions'][boundary:boundary + 16]
        continuous = (act[:, ci] - am[ci]) / ast[ci]
        discrete = (act[:, [6, 13]] > 0.5).long()
        torch.manual_seed(args.seed + chunk)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = wm.rollout(z, discrete[None], continuous[None], prop,
                                time_cache=history, return_time_cache=True,
                                gru_only_boundary=True)
            raw = result.latents[:, 1:16].float() * std + mean
            flat = raw.reshape(15, 108, 16)
            merger = torch.stack(codec.decode(flat, [108] * 15))
            rgb = torch.cat([decoder(batch)[0] for batch in merger.split(4)])
            deep = adapter.features(flat.bfloat16().float(), merger.bfloat16().float())
        assert deep.shape == (15, 3, 108, 2560) and torch.isfinite(deep).all()
        assert torch.isfinite(raw).all() and torch.isfinite(result.proprio).all()
        history = result.time_cache
        assert history.next_kv_cache is None and history.next_rnn_hiddens is not None
        z, prop = result.latents[:, -1:].detach(), result.proprio[:, -1:].detach()
        all_rgb.append(pixels(rgb)); all_latents.append(flat.cpu().float().numpy())
        trace.append({'boundary': boundary, 'predicted_frames': 15, 'carry_gru': True, 'carry_kv': False,
                      'adapter_shape': list(deep.shape)})
        print(f'Completed segment {chunk + 1}/2; GRU carried, KV reset', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = np.concatenate(all_rgb)
    np.savez_compressed(args.output / 'predictions.npz', frame_ids=np.arange(1, 31),
                        rgb=frames, latents=np.concatenate(all_latents))
    images = [Image.fromarray(f) for f in frames]
    images[0].save(args.output / 'prediction.gif', save_all=True, append_images=images[1:], duration=20, loop=0)
    sheet = Image.new('RGB', (448, 248 * 2), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, t in enumerate((15, 30)):
        sheet.paste(Image.open(args.sample.parent / f'reference_{t:03d}.png').convert('RGB'), (0, row * 248 + 24))
        sheet.paste(images[t-1], (224, row * 248 + 24))
        draw.text((4, row * 248 + 5), f'Recorded RGB, frame {t}', fill='black')
        draw.text((228, row * 248 + 5), f'WM decoded RGB, frame {t}', fill='black')
    sheet.save(args.output / 'comparison.png')
    report = {'status': 'passed', 'task': metadata['task'], 'episode_index': metadata['episode_index'],
              'mode': 'recorded-action open-loop, initial latent only; no policy or simulator',
              'frames': len(frames), 'rgb_shape': list(frames.shape), 'seed': args.seed,
              'chunks': trace, 'seconds': time.monotonic() - start,
              'peak_device_memory_gib': torch.cuda.max_memory_allocated() / 1024**3,
              'sample_sha256': metadata['files'][args.sample.name]}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Outputs: {args.output.resolve()}')


if __name__ == '__main__':
    main()
