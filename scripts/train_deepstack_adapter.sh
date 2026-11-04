#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
IRASIM_ROOT="${IRASIM_ROOT:-/mnt/world_foundational_model/chuyao/IRASim}"
exec bash "$SCRIPT_DIR/run_python.sh" "$SCRIPT_DIR/train_deepstack_adapter.py" \
  --cache "${CACHE_ROOT:-$IRASIM_ROOT/cache/robotwin_all50_symmetric_qwen_svae_dim16_50hz}" \
  --raw-data-root "${RAW_DATA_ROOT:-$IRASIM_ROOT/robotdata/robotwin-success-fail-lerobot}" \
  --codec-checkpoint "${CODEC_CHECKPOINT:-$ROOT/outputs/svae_d16_3views/best.pt}" \
  --policy-bundle "${POLICY_BUNDLE:-$IRASIM_ROOT/results/action_expert/tokenworld_qwenoft_success_cam_high_teacher}" \
  "$@"
