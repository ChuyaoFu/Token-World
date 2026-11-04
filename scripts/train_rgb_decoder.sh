#!/usr/bin/env bash
# Standalone RoboTwin cam_high RGB decoder; invoke with bash, including in /bin/sh jobs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/venv/bin/python}"
GPUS="${NUM_GPUS:-8}"
PORT="${MAIN_PROCESS_PORT:-29502}"
CONFIG="$REPO_ROOT/configs/robotwin/decoder/rgb.yaml"
CODEC="$REPO_ROOT/outputs/svae_d16_3views/best.pt"
OUTPUT="$REPO_ROOT/outputs/rgb_decoder_svae_d16_cam_high"
CACHE=""; DATA=""; RESUME=""; DRY_RUN=0; CHECK_ONLY=0; SMOKE=0
EXTRA=()
usage() {
cat <<'HELP'
Usage: bash scripts/train_rgb_decoder.sh --cache PATH --data-root PATH [options]

Inputs and output (relative paths resolve from this checkout):
  --cache PATH              Finalized S-VAE episode cache, same as world-model training
  --data-root PATH          Raw RoboTwin LeRobot data containing RGB videos
  --codec-checkpoint PATH   Frozen S-VAE (default: outputs/svae_d16_3views/best.pt)
  --output PATH             Default: outputs/rgb_decoder_svae_d16_cam_high
  --config PATH             Default: configs/robotwin/decoder/rgb.yaml
  --resume PATH             Resume an RGB decoder checkpoint (not S-VAE / world model)

Runtime:
  --gpus N                  Local PPU ranks (default: 8 / NUM_GPUS)
  --port N                  Distributed port (default: 29502 / MAIN_PROCESS_PORT)
  --python PATH             Default: checkout-relative venv/bin/python
  --batch-size N            GLOBAL trajectory batch, default 32 (8 PPUs: 4 per rank)
  --epochs N                Default 500; cosine LR 3e-4 -> 1e-6
  --num-workers N           Default 4 per rank
  --val-num-workers N       Default 2 per rank
  --keep-feature-cache      Keep decoded features after successful completion
  --check-only              Inspect inputs/config and estimate cache size; no training
  --smoke-test              Tiny one-epoch train/validation/checkpoint run; separate output
  --dry-run                 Print command without SDK/Python, file writes or training
  -h, --help                Show this message

Other trainer options may follow --. This single-node launcher preserves device
visibility from the scheduler and automatically loads the PPU SDK and local venv.
cam_high only; frozen S-VAE -> 108x2560 Qwen features -> 224x224 RGB decoder.
A sequence has 20 frames: the default 4 sequences/rank give 80 RGB images/rank.
First startup materializes a large decoded feature cache under OUTPUT/_latent_cache.
Interrupted runs retain this cache; completed runs remove it unless requested otherwise.
No world-model, policy, or DeepStack interface is changed or trained.
HELP
}
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
 case "$1" in
  --cache) need_value "$@"; CACHE="$2"; shift 2 ;;
  --data-root) need_value "$@"; DATA="$2"; shift 2 ;;
  --codec-checkpoint) need_value "$@"; CODEC="$2"; shift 2 ;;
  --output) need_value "$@"; OUTPUT="$2"; shift 2 ;;
  --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
  --resume) need_value "$@"; RESUME="$2"; shift 2 ;;
  --gpus) need_value "$@"; GPUS="$2"; shift 2 ;;
  --port) need_value "$@"; PORT="$2"; shift 2 ;;
  --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
  --batch-size) need_value "$@"; EXTRA+=(--batch_size "$2"); shift 2 ;;
  --epochs) need_value "$@"; EXTRA+=(--num_epochs "$2"); shift 2 ;;
  --num-workers) need_value "$@"; EXTRA+=(--num_workers "$2"); shift 2 ;;
  --val-num-workers) need_value "$@"; EXTRA+=(--val_num_workers "$2"); shift 2 ;;
  --keep-feature-cache) EXTRA+=(--keep_feature_cache); shift ;;
  --check-only) CHECK_ONLY=1; shift ;;
  --smoke-test) SMOKE=1; shift ;;
  --dry-run) DRY_RUN=1; shift ;;
  -h|--help) usage; exit 0 ;;
  --) shift; EXTRA+=("$@"); break ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
 esac
done
[[ -n "$CACHE" && -n "$DATA" ]] || { usage >&2; exit 2; }
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo '--gpus must be positive' >&2; exit 2; }
[[ "$PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( PORT <= 65535 )) || { echo 'Invalid port' >&2; exit 2; }
if (( SMOKE )); then OUTPUT="${OUTPUT%/}_smoke"; EXTRA+=(--smoke_test); fi
cd -- "$REPO_ROOT"
ARGS=(--config "$CONFIG" --svae_cache_dir "$CACHE" --raw_data_path "$DATA"
 --codec_checkpoint "$CODEC" --checkpoint_dir "$OUTPUT" --latent_cache_dir "$OUTPUT/_latent_cache" --no_wandb)
if [[ -n "$RESUME" ]]; then ARGS+=(--resume "$RESUME"); fi
ARGS+=("${EXTRA[@]}")
if (( CHECK_ONLY )); then
 COMMAND=("$PYTHON_BIN" -m tokenworld.decoder.train "${ARGS[@]}" --check_only --expected_processes "$GPUS")
else
 LAUNCH=(--config_file "$REPO_ROOT/configs/environment/accelerate.yaml"
  --num_machines 1 --machine_rank 0 --num_processes "$GPUS" --mixed_precision bf16
  --dynamo_backend no --num_cpu_threads_per_process 1 --main_process_port "$PORT")
 if (( GPUS > 1 )); then LAUNCH+=(--multi_gpu); fi
 COMMAND=("$PYTHON_BIN" -m accelerate.commands.launch "${LAUNCH[@]}" -m tokenworld.decoder.train "${ARGS[@]}")
fi
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if (( DRY_RUN )); then exit 0; fi
[[ -f "$CONFIG" && -f "$CODEC" && -d "$CACHE" && -d "$DATA" ]] || { echo 'Missing config, codec, cache or raw data directory' >&2; exit 2; }
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then echo 'Resume checkpoint not found' >&2; exit 2; fi
if [[ -z "$RESUME" && "$CHECK_ONLY" == 0 ]]; then
 shopt -s nullglob
 prior=("$OUTPUT"/*.pt)
 (( ${#prior[@]} == 0 )) || { echo 'Output contains checkpoints; use --resume or a new --output' >&2; exit 2; }
fi
source "$SCRIPT_DIR/_environment.sh"
load_training_environment
export PYTHONUNBUFFERED=1
# A shared cache avoids per-node downloads and concurrent first-validation races.
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/outputs/pretrained/torch}"
if (( ! CHECK_ONLY )); then
  "$PYTHON_BIN" -m tokenworld.decoder.train --config "$CONFIG" --prepare_metrics_only
fi
exec "${COMMAND[@]}"
