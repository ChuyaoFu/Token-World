# RoboTwin data contract

The fixed split in `configs/robotwin/splits/seed3047.json` preserves episode
identifiers and seed 3047 from the source workspace. Only its machine-specific
`data_root` was removed. The original file hash is in the source manifest.

The source contains 50 tasks, 2,500 successes and 2,499 failures. Both outcomes
have five validation episodes per task; one failure task (`place_a2b_left`)
has 44 rather than 45 training episodes. Do not silently create a replacement
split to match the nominal paper count of 5,000.

## Raw layout

```text
robotwin-success-fail-lerobot/
  robotwin_success/
    meta/info.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    data/chunk-*/episode_*.parquet
    videos/chunk-*/observation.images.cam_high/episode_*.mp4
    videos/chunk-*/observation.images.cam_left_wrist/episode_*.mp4
    videos/chunk-*/observation.images.cam_right_wrist/episode_*.mp4
  robotwin_fail/
    ...
```

Use metadata path templates rather than assuming one chunk. The recorded
frame rate is 50 Hz. Preserve episode lengths, camera keys, task/outcome
labels, action/state columns and the original episode identifiers.
`tokenworld.data.audit` checks split integrity and referenced-file existence;
it does not decode every video or validate every parquet row.

## Sequence caches

```text
cache/
  cache_spec.json
  cache_plan.json
  index.json
  train_stats.pt
  train/<dataset>/episode_*.pt
  val/<dataset>/episode_*.pt
```

S-VAE training and validation use cam_high by default; `--views all` selects
cam_high and both wrist cameras. The current reproduction uses the three-camera
S-VAE. World-model view selection is configured independently.

Latents are `[time, view, token, channel]`. S-VAE export retains all three
camera grids; the main dynamics config selects `views: [0]` (`cam_high`).
Raw-Qwen export retains cam_high only. Do not concatenate or pool the 108
spatial tokens in these recipes. Export uses posterior means. Normalization
statistics are computed from the training split, with per-view provenance.
The cache's action/state identity check and the dataset's destination-frame
action alignment are preserved; do not shift actions during re-export.

```bash
python -m tokenworld.data.export_svae \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --data-root /path/to/robotwin-success-fail-lerobot \
  --manifest configs/robotwin/splits/seed3047.json \
  --codec-checkpoint /path/to/codec.pt \
  --output data/cache/robotwin_svae_d16
```

After all shards complete, repeat the command with `--finalize-only` to
write the index and train statistics. Use `tokenworld.data.export_raw` for
uncompressed Qwen features (omit `--codec-checkpoint`). Multi-process export
uses `python -m torch.distributed.run ... -m tokenworld.data.export_svae`.
No export is launched as part of release preparation.

## Download the existing dataset

The complete export is already uploaded to [ModelScope: Zellovo/robotwin-success-fail-lerobot](https://modelscope.cn/datasets/Zellovo/robotwin-success-fail-lerobot), whose dataset card declares Apache License 2.0. It is separate from the Token-World model repositories; it does not need to be duplicated inside a model repository.

The release audit found 4,999 episode Parquet files and 14,997 camera videos. All 20,009 remote file paths and sizes match the local source export; SHA-256 also matches for 16 selected files covering metadata and a sample episode. This is a metadata/sample verification, not a claim that every video was decoded. ModelScope's API currently reports gated/login-required access: anonymous listing returned 403, while the owner's authenticated access succeeded. Other users must sign in and meet the repository's access conditions.

```bash
bash scripts/run_python.sh -m pip install 'modelscope-hub==0.4.3'
# Interactive token prompt; do not put a token in a command or repository file.
./venv/bin/modelscope-hub login
bash scripts/run_python.sh - <<'PYCODE'
from modelscope_hub.api import HubApi
HubApi().download_repo(
    'Zellovo/robotwin-success-fail-lerobot', 'dataset', revision='master',
    local_dir='datasets/robotwin-success-fail-lerobot', max_workers=8,
)
PYCODE
export RAW_DATA_ROOT="$PWD/datasets/robotwin-success-fail-lerobot"
bash scripts/run_python.sh -m tokenworld.data.audit \
  --manifest configs/robotwin/splits/seed3047.json \
  --data-root "$RAW_DATA_ROOT" --output outputs/dataset_audit.json
```

Generate the latent cache and `train_stats.pt` with `scripts/reproduce_world_model.sh --stage export` as shown in the root README. Statistics are derived from the fixed training split, not from validation data. Neither caches nor baseline checkpoints need to be uploaded for this release.
