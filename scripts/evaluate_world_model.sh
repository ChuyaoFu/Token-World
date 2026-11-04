#!/usr/bin/env bash
# Evaluate a RoboTwin S-VAE world model in the release PPU environment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_python.sh" -m tokenworld.evaluation.open_loop \
  --profile robotwin --latent-method symmetric_qwen --views 0 \
  --reference raw_teacher --split val --selection balanced_task_outcome \
  --max-episodes 100 --max-steps 0 --initial-context-frames 1 \
  --sampling-steps 16 --seed 3047 "$@"
