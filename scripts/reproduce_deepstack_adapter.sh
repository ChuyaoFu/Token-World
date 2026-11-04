#!/usr/bin/env bash
# Single-node PPU job: N-rank extraction -> one-PPU train/eval -> owned feature cleanup.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
IRASIM_ROOT="${IRASIM_ROOT:-$ROOT/../IRASim}"
GPUS=8
STAGE=all
KEEP=0
DRY=0
ARGS=()
usage() {
  cat <<'HELP'
Usage: bash scripts/reproduce_deepstack_adapter.sh --output PATH [options]

--gpus N          Export workers on this node (default 8). Training uses cuda:0.
--stage STAGE     all (default), prepare, export, train, cleanup
--keep-features   Preserve this run's temporary feature files after success
--dry-run         Print the pipeline without loading models or writing files

Other arguments go to deepstack_pipeline.py. Defaults: all 50 tasks, all official
train/val episodes, cam_high, 8 train frames/episode, 4 validation frames/episode,
30 epochs, batch 32, LR 3e-4. --steps N overrides epoch count. --workers N controls
CPU data loading. The existing train/val split is preserved. This is a single-node
8-PPU launcher, not an 8-node launcher. Use bash explicitly in /bin/sh job runners.

On success, cleanup removes only manifest-listed feature shards from this run,
after checkpoint integrity checks and full final action evaluation. Failures keep
features and checkpoints. Re-run the identical command to reuse export and resume
training. Use --stage cleanup to retry an interrupted cleanup alone.
HELP
}
while (($#)); do
 case "$1" in
  --gpus|--stage)
   [[ $# -ge 2 ]] || { usage; exit 2; }
   if [[ "$1" == --gpus ]]; then GPUS="$2"; else STAGE="$2"; fi
   shift 2 ;;
  --keep-features) KEEP=1; shift ;;
  --dry-run) DRY=1; shift ;;
  -h|--help) usage; exit 0 ;;
  *) ARGS+=("$1"); shift ;;
 esac
done
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo '--gpus must be positive' >&2; exit 2; }
[[ "$STAGE" =~ ^(all|prepare|export|train|cleanup)$ ]] || { echo 'Invalid stage' >&2; exit 2; }
cd -- "$ROOT"
BASE=("$SCRIPT_DIR/deepstack_pipeline.py"
 --cache "${CACHE_ROOT:-$IRASIM_ROOT/cache/robotwin_all50_symmetric_qwen_svae_dim16_50hz}"
 --raw-data-root "${RAW_DATA_ROOT:-$IRASIM_ROOT/robotdata/robotwin-success-fail-lerobot}"
 --codec-checkpoint "${CODEC_CHECKPOINT:-$ROOT/outputs/svae_d16_3views/best.pt}"
 --policy-bundle "${POLICY_BUNDLE:-$IRASIM_ROOT/results/action_expert/tokenworld_qwenoft_success_cam_high_teacher}")
run() { printf 'Command: '; printf '%q ' "$@"; printf '\n'; if (( ! DRY )); then "$@"; fi; }
run env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT bash "$SCRIPT_DIR/run_python.sh" "${BASE[@]}" "${ARGS[@]}" --stage prepare
if [[ "$STAGE" == all || "$STAGE" == export ]]; then
 run env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT bash "$SCRIPT_DIR/run_python.sh" -m torch.distributed.run --standalone "--nproc_per_node=$GPUS" \
  "${BASE[@]}" "${ARGS[@]}" --stage export
 run env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT bash "$SCRIPT_DIR/run_python.sh" "${BASE[@]}" "${ARGS[@]}" --stage finalize
fi
if [[ "$STAGE" == all || "$STAGE" == train ]]; then
 run env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT bash "$SCRIPT_DIR/run_python.sh" "${BASE[@]}" "${ARGS[@]}" --stage train
 if (( ! KEEP )); then run env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT bash "$SCRIPT_DIR/run_python.sh" "${BASE[@]}" "${ARGS[@]}" --stage cleanup; fi
fi
if [[ "$STAGE" == cleanup ]]; then
 run env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT bash "$SCRIPT_DIR/run_python.sh" "${BASE[@]}" "${ARGS[@]}" --stage cleanup
fi
