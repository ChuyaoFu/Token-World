#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd)
release=$(cd "$scripts/../.." && pwd)
src="$release/../IRASim/results/action_expert/tokenworld_qwenoft_success_cam_high_teacher/code/robotwin"
runtime=${ROBOTWIN_SIM_RUNTIME:-"$(dirname "$(dirname "$release")")/zgy/robotwin_closedloop_20261021"}
trap 'rc=$?; echo "Preparation failed at line $LINENO (exit $rc): $BASH_COMMAND" >&2' ERR
stage=${TOKENWORLD_SIM_STAGE:?set stage directory on a node-local disk}
mkdir -p "$(dirname -- "$stage")"
exec 9>"$stage.lock"
echo "Waiting for local asset lock: $stage.lock"
flock -w 900 9 || { echo "Timed out waiting for $stage.lock" >&2; exit 1; }
for dependency in "$runtime/sim-env/bin/python" "$runtime/mesa-prefix/usr/share/vulkan/icd.d/lvp_icd.json" "$src/task_config/demo_clean.yml"; do
    [[ -r "$dependency" ]] || { echo "Missing simulator dependency: $dependency" >&2; exit 1; }
done
if [[ -f "$stage/ready-v1" ]]; then echo "Using prepared stage: $stage"; exit 0; fi
if [[ -e "$stage" ]]; then echo "Incomplete stage at $stage. Set TOKENWORLD_SIM_STAGE to a new path." >&2; exit 1; fi
stage_tmp=$(mktemp -d "$stage.build.XXXXXX")
export TOKENWORLD_SIM_STAGE="$stage_tmp"
mkdir -p "$stage_tmp/sim/assets/objects"
echo "Copying RoboTwin source from $src to $stage_tmp"
cp -r "$src/envs" "$src/task_config" "$stage_tmp/sim/"
metadata="$runtime/RoboTwin/assets/objects"
for name in objaverse/list.json same.json; do
    [[ -r "$metadata/$name" ]] || { echo "Missing RoboTwin metadata: $metadata/$name" >&2; exit 1; }
    mkdir -p "$stage_tmp/sim/assets/objects/$(dirname "$name")"
    cp "$metadata/$name" "$stage_tmp/sim/assets/objects/$name"
done
ln -s assets/embodiments/aloha-agilex "$stage_tmp/sim/aloha_maniskill_sim"
echo "Downloading and verifying task assets"
"$release/venv/bin/python" -u "$scripts/fetch.py"
echo "Downloading and verifying the robot model"
"$release/venv/bin/python" -u "$scripts/embodiment.py"
touch "$stage_tmp/ready-v1"
mv "$stage_tmp" "$stage"
