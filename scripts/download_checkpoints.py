"""Download the pinned RoboTwin checkpoints and verify their original bytes."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('huggingface', 'modelscope'), default='huggingface')
    parser.add_argument('--repo-id', help='Public model repository owner/name')
    parser.add_argument('--revision', default='main', help='Published revision or pinned commit')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and not args.repo_id:
        parser.error('--repo-id is required to download checkpoints')
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'configs/checkpoints.json').read_text())
    for entry in manifest['files']:
        path = args.root / entry['path']
        if not path.is_file() and not args.verify_only:
            if args.provider == 'huggingface':
                from huggingface_hub import hf_hub_download
                hf_hub_download(args.repo_id, entry['path'], revision=args.revision, local_dir=args.root)
            else:
                from modelscope_hub.api import HubApi
                HubApi().download_file(args.repo_id, 'model', entry['path'], revision=args.revision,
                                       local_dir=args.root, expected_sha256=entry['sha256'])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise RuntimeError(f'Checkpoint mismatch at {path}. Move this file aside before retrying.')
        print(f"Verified {entry['component']}: {path}", flush=True)


if __name__ == '__main__':
    main()
