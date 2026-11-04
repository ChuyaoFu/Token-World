# Training recipes

All artifact locations are arguments or YAML settings. W&B is off by default;
if enabled, supply credentials through your environment, never source files.
The recipes below were migrated from the research workspace. A clean-machine
GPU training run and paper-checkpoint mapping are still pending.

## Training scripts

Use `bash scripts/train_svae.sh --help` or
`bash scripts/train_world_model.sh --help` for the supported launch wrappers.
See [script usage](../scripts/README.md) for paths, process counts, dry runs and
resume examples. Export/finalize the S-VAE cache between these two stages.
The module commands below remain available for direct use.

## Semantic VAE

Training and validation use **cam_high only**. Frames are decoded online;
the frozen Qwen teacher produces 108x2560 features and the default codec
compresses them to 108x16. No wrist-camera videos or feature cache are
required for this stage. `--batch-size` counts cam_high images per GPU.
The training plan and checkpoints record `video_keys`; resume rejects
checkpoints with a different or unspecified camera protocol.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m tokenworld.codec.train \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --data-root /path/to/robotwin-success-fail-lerobot \
  --manifest configs/robotwin/splits/seed3047.json \
  --output outputs/svae_d16 --token-dim 16 \
  --latent-grid-size 9 12 --expected-tasks 50 --expected-world-size 8 \
  --require-all-episodes-in-split
```

The codec implementation is included and does not require importing IRASim
or the whole StarVLA policy stack. The frozen Qwen model is a separate asset.
Export a sequence cache after codec training as described in `data.md`.

## Dynamics

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m tokenworld.dynamics.train \
  --config configs/robotwin/dynamics/svae_d16.yaml \
  --data_path /path/to/svae_cache \
  --checkpoint_folder outputs/world_model
```

The main workspace recipe uses 768 model dimensions, 12 blocks, 108 spatial
tokens, a 9 x 12 grid, four history frames, 16 prediction frames and a
100,000-step training budget. `raw_qwen.yaml` uses 2560 feature channels.
Dimension configs for 8, 32 and 48 channels are derived recipes, not recovered
proof of the exact Table III runs. Resume with `--resume_from` explicitly.

## RGB decoder

Set `data.codec_checkpoint` in `configs/robotwin/decoder/rgb.yaml` to the
selected codec before running:

```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
  -m tokenworld.decoder.train \
  --config configs/robotwin/decoder/rgb.yaml \
  --raw_data_path /path/to/robotwin-success-fail-lerobot \
  --svae_cache_dir /path/to/svae_cache \
  --latent_cache_dir outputs/decoder_feature_cache \
  --checkpoint_dir outputs/rgb_decoder --no_wandb
```

The RGB decoder consumes S-VAE-decoded 2560-channel Qwen features and renders
224 x 224 frames. Its temporary feature cache must be distinct from the source
S-VAE cache. It is used for video inspection/manual success assessment, not
policy feedback. Training reconstruction previews are not paper plots.
