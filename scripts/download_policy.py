#!/usr/bin/env python3
"""Download and verify the optional RoboTwin QwenOFT policy bundle."""
import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from download_checkpoints import sha256

ROOT = Path(__file__).resolve().parents[1]


def verify_support(bundle, verified_weight):
    for line in (bundle / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = bundle / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path != verified_weight and sha256(path) != digest:
            raise RuntimeError(f'Policy support file mismatch: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('huggingface', 'modelscope'), default='huggingface')
    parser.add_argument('--repo-id')
    parser.add_argument('--revision')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    repo = args.repo_id or {'huggingface': 'ChuyaoFu/Token-World', 'modelscope': 'Zellovo/Token-World'}[args.provider]
    revision = args.revision or ('main' if args.provider == 'huggingface' else 'master')
    manifest = json.loads((ROOT / 'configs/policy.json').read_text())
    root = args.root.expanduser().resolve()
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file() and not args.verify_only:
            if args.provider == 'huggingface':
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo, entry['path'], revision=revision, local_dir=root, token=False)
            else:
                from modelscope_hub.api import HubApi
                HubApi(token='').download_file(repo, 'model', entry['path'], revision=revision,
                                               local_dir=root, expected_sha256=entry['sha256'])
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise RuntimeError(f'Missing or mismatched policy artifact: {path}')
        print(f"Verified {entry['component']}: {path}", flush=True)
    bundle = root / manifest['bundle_root']
    weight = root / next(e['path'] for e in manifest['files'] if e['component'] == 'policy')
    if not args.verify_only:
        archive = root / next(e['path'] for e in manifest['files'] if e['component'] == 'policy_support')
        with tempfile.TemporaryDirectory(prefix='.policy-unpack-', dir=root) as temporary:
            temporary = Path(temporary)
            with tarfile.open(archive, 'r:gz') as tar:
                for member in tar.getmembers():
                    path = Path(member.name)
                    if (not member.isfile() or path.is_absolute() or '..' in path.parts
                            or not path.is_relative_to(manifest['bundle_root'])):
                        raise RuntimeError(f'Unexpected archive member: {member.name}')
                tar.extractall(temporary, filter='data')
            sources = [p for p in temporary.rglob('*') if p.is_file()]
            for source in sources:
                target = root / source.relative_to(temporary)
                if target.exists() and (target.is_symlink() or sha256(target) != sha256(source)):
                    raise RuntimeError(f'Refusing to overwrite modified policy support file: {target}')
            for source in sources:
                target = root / source.relative_to(temporary)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    verify_support(bundle, weight)
    print(f'Policy bundle ready. Set POLICY_BUNDLE={bundle}', flush=True)


if __name__ == '__main__':
    main()
