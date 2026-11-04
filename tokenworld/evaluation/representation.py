"""Measure LDS and DCT-based SEC on a RoboTwin S-VAE latent cache."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
import torch.nn.functional as F


def dct_matrix(n, device):
    i = torch.arange(n, dtype=torch.float64, device=device)[:, None]
    k = torch.arange(n, dtype=torch.float64, device=device)[None, :]
    mat = torch.cos(math.pi / n * (i + 0.5) * k)
    mat[:, 0] *= math.sqrt(1.0 / n)
    mat[:, 1:] *= math.sqrt(2.0 / n)
    return mat


@torch.no_grad()
def sec(latent, h, w, rho):
    # latent: [B,H*W,D]. SEC_rho is high-frequency DCT energy fraction.
    x = latent.reshape(-1, h, w, latent.shape[-1]).double()
    ch, cw = dct_matrix(h, x.device), dct_matrix(w, x.device)
    coeff = torch.einsum('mh,bmwd->bhwd', ch, x)
    coeff = torch.einsum('wn,bhwd->bhnd', cw, coeff)
    energy = coeff.square().mean(dim=-1)
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
    high = (yy + xx).double() > float(rho) * float(h + w - 2)
    return energy[:, high].sum(dim=1) / energy.sum(dim=(1, 2)).clamp_min(1e-12)


@torch.no_grad()
def lds(latent, h, w, far_dist):
    # Exact iREPA LDS: mean 4-neighbor cosine minus mean cosine for Manhattan >= far_dist.
    x = F.normalize(latent.float(), dim=-1, eps=1e-8).reshape(-1, h, w, latent.shape[-1])
    local = torch.cat([
        (x[:, 1:] * x[:, :-1]).sum(-1).flatten(1),
        (x[:, :, 1:] * x[:, :, :-1]).sum(-1).flatten(1),
    ], dim=1).mean(dim=1)
    flat = x.reshape(-1, h * w, x.shape[-1])
    gram = torch.einsum('btd,bsd->bts', flat, flat)
    coords = torch.stack(torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij'), dim=-1).reshape(-1, 2)
    dist = (coords[:, None] - coords[None, :]).abs().sum(dim=-1)
    far = (dist >= far_dist) & (dist > 0)
    return local - gram[:, far].mean(dim=1)


def summarize(values):
    x = torch.cat(values).double()
    return {'mean': float(x.mean()), 'std': float(x.std(unbiased=False)), 'min': float(x.min()), 'max': float(x.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--split', choices=('train','val'), default='val')
    ap.add_argument('--view', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--sec-rho', type=float, default=0.5)
    ap.add_argument('--lds-far-dist', type=int, default=6)
    args = ap.parse_args()
    spec = json.loads((args.cache / 'cache_spec.json').read_text())
    index = json.loads((args.cache / 'index.json').read_text())
    latent_method = spec['latent_method']
    latent_spec = spec['latent_specs'][latent_method]['shape_per_frame']
    views, tokens, dim = map(int, latent_spec)
    if args.view < 0 or args.view >= views:
        raise ValueError(f'view {args.view} outside cache view count {views}')
    grid = list(map(int, spec['codec_config'].get('latent_grid_size', spec['codec_config']['grid_size'])))
    h, w = grid
    if h * w != tokens:
        if views > 1 and h * w == views * tokens and w % views == 0:
            w //= views
            grid = [h, w]
        else:
            raise ValueError(f'grid {grid} does not match token count {tokens}')
    rows = [r for r in index['records'] if r['split'] == args.split]
    sec_values, lds_values = [], []
    frame_count = 0
    for row in rows:
        payload = torch.load(args.cache / row['path'], map_location='cpu', weights_only=True)
        z = payload['latents'][latent_method][:, args.view].float()
        for start in range(0, z.shape[0], args.batch_size):
            batch = z[start:start + args.batch_size]
            sec_values.append(sec(batch, h, w, args.sec_rho).cpu())
            lds_values.append(lds(batch, h, w, args.lds_far_dist).cpu())
        frame_count += int(z.shape[0])
    report = {
        'version': 2,
        'cache': str(args.cache.resolve()),
        'latent_method': latent_method,
        'latent_shape_per_frame': latent_spec,
        'latent_dim': dim,
        'split': args.split,
        'view_index': args.view,
        'video_key': spec['video_keys'][args.view],
        'frames': frame_count,
        'grid': [h, w],
        'SEC': {
            'definition': 'fraction of orthonormal 2D-DCT energy with u+v > rho*(H+W-2)',
            'rho': args.sec_rho,
            **summarize(sec_values),
        },
        'LDS': {
            'definition': 'mean 4-neighbor cosine minus mean cosine for Manhattan distance >= far_dist',
            'far_dist': args.lds_far_dist,
            **summarize(lds_values),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
