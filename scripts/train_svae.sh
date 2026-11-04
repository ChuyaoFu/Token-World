#!/usr/bin/env bash
# Train the RoboTwin semantic VAE using an explicit camera preset.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/venv/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
MODEL=""
DATA_ROOT=""
MANIFEST="$REPO_ROOT/configs/robotwin/splits/seed3047.json"
OUTPUT=""
VIEWS="cam_high"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: train_svae.sh --model PATH --data-root PATH [options] [trainer arguments]

Required:
  --model PATH       Local Qwen3-VL model directory (108 x 2560 visual features)
  --data-root PATH   RoboTwin success/failure LeRobot dataset root

Options:
  --gpus N           Number of local accelerator processes (default: 8 / NUM_GPUS)
  --python PATH      Python executable (default: repo venv/bin/python / PYTHON_BIN)
  --manifest PATH    Split manifest (default: configs/robotwin/splits/seed3047.json)
  --views PRESET     cam_high (default) or all (high + left/right wrists)
  --output PATH      Output: outputs/svae_d16 or outputs/svae_d16_3views
  --resume PATH      Resume an interrupted run from resume.pt
  --dry-run          Print the command without loading models or starting training
  -h, --help         Show this help

Other arguments are forwarded to tokenworld.codec.train; for example:
  --epochs 1 --batch-size 8 --token-dim 16 --learning-rate 1e-4
  --plan-only        Inspect the actual data/step plan without training
  --no-auto-resume   Disable automatic resume from the output directory

--views applies to both training and validation. Defaults: d=16, grid=9x12,
seed=3047, all training frames, up to 8 validation frames per episode.
Relative paths are resolved from the repository root. The PPU SDK and repo venv
are loaded automatically. Install once with: bash scripts/setup_env.sh
EOF
}
need_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
    printf 'Missing value for %s\n' "$1" >&2
    exit 2
  fi
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) need_value "$@"; MODEL="$2"; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --manifest) need_value "$@"; MANIFEST="$2"; shift 2 ;;
    --output) need_value "$@"; OUTPUT="$2"; shift 2 ;;
    --views) need_value "$@"; VIEWS="$2"; shift 2 ;;
    --gpus) need_value "$@"; NUM_GPUS="$2"; shift 2 ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--gpus must be a positive integer' >&2; exit 2; }
[[ -n "$MODEL" && -n "$DATA_ROOT" ]] || { usage >&2; exit 2; }

case "$VIEWS" in
  cam_high) OUTPUT="${OUTPUT:-$REPO_ROOT/outputs/svae_d16}" ;;
  all) OUTPUT="${OUTPUT:-$REPO_ROOT/outputs/svae_d16_3views}" ;;
  *) printf '%s\n' '--views must be cam_high or all' >&2; exit 2 ;;
esac
cd -- "$REPO_ROOT"
# Explicit defaults describe the release recipe; trailing trainer args may override them.
COMMAND=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc_per_node=$NUM_GPUS"
  -m tokenworld.codec.train
  --model "$MODEL" --data-root "$DATA_ROOT" --manifest "$MANIFEST" --output "$OUTPUT"
  --views "$VIEWS" --token-dim 16 --latent-grid-size 9 12 --seed 3047
  --epochs 1 --batch-size 8 --eval-batch-size 8 --learning-rate 1e-4 --kl-weight 1e-6
  --train-frames-per-episode 0 --val-frames-per-episode 8
  --expected-tasks 50 --require-all-episodes-in-split
  --save-every-steps 5000 --save-every-epoch
  "${EXTRA_ARGS[@]}" --expected-world-size "$NUM_GPUS")
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
source "$SCRIPT_DIR/_environment.sh"
load_training_environment
exec "${COMMAND[@]}"
