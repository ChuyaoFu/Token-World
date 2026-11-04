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

The public download location, data license text and archive checksums remain
to be finalized. The existing dataset card only declares Apache License 2.0;
the code license must not be substituted for a dataset license decision.
