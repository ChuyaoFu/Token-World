"""Prepare all RoboTwin objects in shared storage and a private simulator stage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import requests

REVISION = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'
URL = f'https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0/resolve/{REVISION}/objects.zip'
SIZE = 3737778549
BLOCK = 8 * 1024 * 1024


def prepare(base, stage, cache):
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = cache / 'ready.json'
        if not ready.exists():
            chunks = cache / 'chunks'
            chunks.mkdir(exist_ok=True)

            def fetch(start):
                end = min(SIZE, start + BLOCK) - 1
                dst = chunks / str(start)
                if dst.exists() and dst.stat().st_size == end - start + 1:
                    return dst
                for attempt in range(5):
                    try:
                        response = requests.get(URL + f'?range_start={start}&end={end}',
                                                headers={'Range': f'bytes={start}-{end}'}, timeout=(20, 180))
                        response.raise_for_status()
                        assert response.status_code == 206
                        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                        assert len(response.content) == end - start + 1
                        tmp = dst.with_suffix('.tmp')
                        tmp.write_bytes(response.content)
                        tmp.replace(dst)
                        return dst
                    except Exception:
                        if attempt == 4:
                            raise
                        time.sleep(2)

            archive = cache / 'objects.zip'
            with ThreadPoolExecutor(8) as pool, archive.open('wb') as out:
                for i, chunk in enumerate(pool.map(fetch, range(0, SIZE, BLOCK))):
                    with chunk.open('rb') as src:
                        shutil.copyfileobj(src, out)
                    if i % 16 == 0:
                        print(f'All-task assets: {i + 1}/{(SIZE + BLOCK - 1) // BLOCK} blocks', flush=True)
            build = cache / 'extract'
            build.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    if info.is_dir() or '/._' in info.filename or not info.filename.startswith('objects/'):
                        continue
                    target = (build / info.filename).resolve()
                    if (build / 'objects').resolve() not in target.parents:
                        raise ValueError(f'Unexpected archive member: {info.filename}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Reading to EOF verifies the archive member CRC.
                    with z.open(info) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
            # Metadata used by random object sampling is not present in every archive.
            for name in ('same.json', 'objaverse/list.json'):
                dst = build / 'objects' / name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(base / 'assets/objects' / name, dst)
            (build / 'objects').replace(cache / 'objects')
            ready.write_text(json.dumps(dict(url=URL, revision=REVISION, bytes=SIZE,
                                            integrity='ZIP member CRC checked', complete=True), indent=2))
            shutil.rmtree(chunks)
            archive.unlink()
    stage.parent.mkdir(parents=True, exist_ok=True)
    with stage.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (stage / 'ready.json').exists():
            saved = json.loads((stage / 'ready.json').read_text())
            assert saved == dict(base=str(base), objects=str(cache / 'objects'))
            return
        if stage.exists():
            raise ValueError(f'Incomplete stage {stage}; remove it or choose another stage')
        build = stage.with_name(stage.name + f'.build.{os.getpid()}')
        build.mkdir()
        shutil.copytree(base / 'envs', build / 'envs', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(base / 'task_config', build / 'task_config')
        (build / 'assets').mkdir()
        for item in (base / 'assets').iterdir():
            (build / 'assets' / item.name).symlink_to(cache / 'objects' if item.name == 'objects' else item.resolve())
        (build / 'aloha_maniskill_sim').symlink_to('assets/embodiments/aloha-agilex')
        (build / 'ready.json').write_text(json.dumps(dict(base=str(base), objects=str(cache / 'objects'))))
        build.replace(stage)
    print(f'All-task simulator ready: {stage}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    a = p.parse_args()
    prepare(a.base.resolve(), a.stage.resolve(), a.cache.resolve())
