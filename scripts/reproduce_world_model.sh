#!/usr/bin/env bash
# Reproduce RoboTwin world-model training from an S-VAE checkpoint or ready cache.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STAGE=all
MODEL=""
DATA_ROOT=""
CODEC="$REPO_ROOT/outputs/svae_d16_3views/best.pt"
CACHE="$REPO_ROOT/outputs/cache/robotwin_svae_d16_3views_50hz"
MANIFEST="$REPO_ROOT/configs/robotwin/splits/seed3047.json"
CONFIG="$REPO_ROOT/configs/robotwin/dynamics/svae_d16.yaml"
OUTPUT="$REPO_ROOT/outputs/world_model_svae_d16_cam_high_100k"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/venv/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
PORT="${MAIN_PROCESS_PORT:-29500}"
IMAGE_BATCH_SIZE=12
RESUME=""
DRY_RUN=0
usage() {
  cat <<'HELP'
Usage: bash scripts/reproduce_world_model.sh [options]

Stages:
  --stage all       Export -> finalize -> check -> train (default)
  --stage export    Export -> finalize -> check, without training
  --stage check     Check an existing finalized cache, without using devices
  --stage train     Check an existing finalized cache -> train

Inputs:
  --model PATH             Qwen3-VL model directory (required for all/export)
  --data-root PATH         RoboTwin LeRobot dataset (required for all/export)
  --codec-checkpoint PATH  S-VAE weights (default: outputs/svae_d16_3views/best.pt)
  --cache PATH             Cache input/output (default: outputs/cache/robotwin_svae_d16_3views_50hz)
  --manifest PATH          Episode split (default: configs/robotwin/splits/seed3047.json)
  --config PATH            World-model YAML (default: configs/robotwin/dynamics/svae_d16.yaml)
  --output PATH            Training output (default: outputs/world_model_svae_d16_cam_high_100k)

Runtime:
  --gpus N                 Local PPU processes (default: 8 / NUM_GPUS)
  --image-batch-size N      Qwen export image batch; multiple of 3 (default: 12)
  --python PATH            Override repo venv/bin/python / PYTHON_BIN
  --port N                 World-model distributed port (default: 29500)
  --resume PATH            World-model checkpoint (use with --stage train)
  --dry-run                Print the workflow without loading SDK/Python or writing files
  -h, --help               Show usage

Paths are relative to the repository root, regardless of the caller's directory.
The PPU SDK and repo venv are loaded automatically. Run setup_env.sh once first.
Export uses all 3 cameras, posterior means and 50-Hz frames. The default world
model uses cam_high only: 4 history + 16 prediction frames, batch 32 per process,
BF16, learning rate 6e-5 and 100,000 optimizer steps. Eight PPUs give global
batch 256. This is a single-node launcher; preserve scheduler device visibility.

Reuse an existing cache via --stage train --cache PATH. To resume an interrupted
export, repeat the same all/export invocation with unchanged inputs. Existing
shards are reused; this wrapper never passes --overwrite. W&B is disabled by
the default YAML. Model weights/data are not downloaded by this script.
HELP
}
need_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) need_value "$@"; STAGE="$2"; shift 2 ;;
    --model) need_value "$@"; MODEL="$2"; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --codec-checkpoint) need_value "$@"; CODEC="$2"; shift 2 ;;
    --cache) need_value "$@"; CACHE="$2"; shift 2 ;;
    --manifest) need_value "$@"; MANIFEST="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
    --output) need_value "$@"; OUTPUT="$2"; shift 2 ;;
    --gpus) need_value "$@"; NUM_GPUS="$2"; shift 2 ;;
    --image-batch-size) need_value "$@"; IMAGE_BATCH_SIZE="$2"; shift 2 ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --port) need_value "$@"; PORT="$2"; shift 2 ;;
    --resume) need_value "$@"; RESUME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$STAGE" =~ ^(all|export|check|train)$ ]] || { echo 'Invalid --stage' >&2; exit 2; }
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo '--gpus must be positive' >&2; exit 2; }
[[ "$IMAGE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] && (( IMAGE_BATCH_SIZE % 3 == 0 )) || { echo '--image-batch-size must be a positive multiple of 3' >&2; exit 2; }
[[ "$PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( PORT <= 65535 )) || { echo '--port must be between 1 and 65535' >&2; exit 2; }
if [[ -n "$RESUME" && "$STAGE" != train ]]; then echo '--resume requires --stage train' >&2; exit 2; fi
if [[ "$STAGE" == all || "$STAGE" == export ]]; then
  [[ -n "$MODEL" && -n "$DATA_ROOT" ]] || { echo '--model and --data-root are required for export' >&2; exit 2; }
fi
cd -- "$REPO_ROOT"
export PYTHON_BIN
run() {
  printf 'Command: '; printf '%q ' "$@"; printf '\n'
  if [[ "$DRY_RUN" == 0 ]]; then "$@"; fi
}
if [[ "$DRY_RUN" == 0 ]]; then
  [[ -f "$CONFIG" && -f "$MANIFEST" ]] || { echo 'Config or manifest does not exist' >&2; exit 2; }
  if [[ "$STAGE" == all || "$STAGE" == export ]]; then
    [[ -d "$MODEL" && -d "$DATA_ROOT" && -f "$CODEC" ]] || { echo 'Model, data root or codec checkpoint does not exist' >&2; exit 2; }
  fi
  if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then echo 'Resume checkpoint does not exist' >&2; exit 2; fi
  if [[ ( "$STAGE" == all || "$STAGE" == train ) && -z "$RESUME" ]]; then
    shopt -s nullglob
    prior_checkpoints=("$OUTPUT"/*.pt)
    (( ${#prior_checkpoints[@]} == 0 )) || { echo 'Output contains checkpoints; choose a new --output or use --stage train --resume PATH' >&2; exit 2; }
  fi
fi
if [[ "$STAGE" == all || "$STAGE" == export ]]; then
  EXPORT_ARGS=(--model "$MODEL" --data-root "$DATA_ROOT" --manifest "$MANIFEST"
    --codec-checkpoint "$CODEC" --output "$CACHE" --image-batch-size "$IMAGE_BATCH_SIZE"
    --image-height 288 --image-width 384 --export-frame-stride 1 --expected-task-count 50)
  run bash "$SCRIPT_DIR/run_python.sh" -m torch.distributed.run --standalone "--nproc_per_node=$NUM_GPUS" \
    -m tokenworld.data.export_svae "${EXPORT_ARGS[@]}"
  run bash "$SCRIPT_DIR/run_python.sh" -m tokenworld.data.export_svae "${EXPORT_ARGS[@]}" --finalize-only
fi
run bash "$SCRIPT_DIR/run_python.sh" "$SCRIPT_DIR/check_world_model_cache.py" \
  --cache "$CACHE" --config "$CONFIG" --manifest "$MANIFEST"
if [[ "$STAGE" == all || "$STAGE" == train ]]; then
  TRAIN_ARGS=(--cache "$CACHE" --config "$CONFIG" --gpus "$NUM_GPUS"
    --python "$PYTHON_BIN" --output "$OUTPUT" --port "$PORT")
  if [[ -n "$RESUME" ]]; then TRAIN_ARGS+=(--resume "$RESUME"); fi
  if [[ "$DRY_RUN" == 1 ]]; then
    run bash "$SCRIPT_DIR/train_world_model.sh" "${TRAIN_ARGS[@]}"
  else
    exec bash "$SCRIPT_DIR/train_world_model.sh" "${TRAIN_ARGS[@]}"
  fi
fi
