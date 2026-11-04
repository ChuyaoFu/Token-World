#!/usr/bin/env bash
# Train the RoboTwin world model on a previously exported semantic-token cache.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/venv/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
CACHE=""
CONFIG="$REPO_ROOT/configs/robotwin/dynamics/svae_d16.yaml"
OUTPUT="$REPO_ROOT/outputs/world_model_svae_d16"
RESUME=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: train_world_model.sh --cache PATH [options] [trainer arguments]

Required:
  --cache PATH       Finalized S-VAE sequence cache, not raw videos or a codec .pt

Options:
  --gpus N           Number of local accelerator processes (default: 8 / NUM_GPUS)
  --python PATH      Python executable (default: repo venv/bin/python / PYTHON_BIN)
  --config PATH      Dynamics YAML (default: configs/robotwin/dynamics/svae_d16.yaml)
  --output PATH      Checkpoint directory (default: outputs/world_model_svae_d16)
  --resume PATH      Explicit world-model checkpoint to resume
  --port N           Distributed port (default: 29500 / MAIN_PROCESS_PORT)
  --dry-run          Print the command without loading data or starting training
  -h, --help         Show this help

Other arguments are forwarded to tokenworld.dynamics.train; for example:
  --batch_size 16 --learning_rate 6e-5

The default YAML selects cam_high (view 0), d=16, 108 tokens, four history
frames, 16 prediction frames and 100,000 optimizer steps; W&B/RGB decoder
training are disabled. Edit the YAML to change max_train_steps. --num_epochs
alone does not reduce the budget while max_train_steps is set.
Relative paths are resolved from the repository root. The PPU SDK and repo venv
are loaded automatically. Install once with: bash scripts/setup_env.sh
No cache export is launched.
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
    --cache) need_value "$@"; CACHE="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG="$2"; shift 2 ;;
    --output) need_value "$@"; OUTPUT="$2"; shift 2 ;;
    --resume) need_value "$@"; RESUME="$2"; shift 2 ;;
    --gpus) need_value "$@"; NUM_GPUS="$2"; shift 2 ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --port) need_value "$@"; MAIN_PROCESS_PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--gpus must be a positive integer' >&2; exit 2; }
[[ "$MAIN_PROCESS_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( MAIN_PROCESS_PORT <= 65535 )) || { printf '%s\n' '--port must be an integer between 1 and 65535' >&2; exit 2; }
[[ -n "$CACHE" ]] || { usage >&2; exit 2; }
cd -- "$REPO_ROOT"
[[ -f "$CONFIG" ]] || { printf 'Config does not exist: %s\n' "$CONFIG" >&2; exit 2; }

# Explicit topology avoids silently using a personal Accelerate launch config.
LAUNCH_ARGS=(--config_file "$REPO_ROOT/configs/environment/accelerate.yaml"
  --num_machines 1 --machine_rank 0 --num_processes "$NUM_GPUS"
  --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1
  --main_process_port "$MAIN_PROCESS_PORT")
if (( NUM_GPUS > 1 )); then LAUNCH_ARGS+=(--multi_gpu); fi
COMMAND=("$PYTHON_BIN" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}"
  -m tokenworld.dynamics.train --config "$CONFIG"
  --data_path "$CACHE" --checkpoint_folder "$OUTPUT")
if [[ -n "$RESUME" ]]; then COMMAND+=(--resume_from "$RESUME"); fi
COMMAND+=("${EXTRA_ARGS[@]}")
printf 'Command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
source "$SCRIPT_DIR/_environment.sh"
load_training_environment
exec "${COMMAND[@]}"
