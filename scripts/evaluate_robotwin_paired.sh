#!/usr/bin/env bash
# Single node, one independent model worker per card, colocated CPU physics.
set -euo pipefail
release=$(cd "$(dirname "$0")/.." && pwd)
helper="$release/scripts/robotwin_sim"
cards=${1:-8}
seeds=${2:-25}
first=${3:-100000}
mode=${ROBOTWIN_CACHE_MODE:-gru}
case "$mode" in gru|full|reset) ;; *) echo "Invalid ROBOTWIN_CACHE_MODE: $mode" >&2; exit 2;; esac
out=${4:-"$release/outputs/robotwin_wm_pair_${mode}_$(date -u +%Y%m%dT%H%M%SZ)"}
[[ "$cards" =~ ^[1-9][0-9]*$ && "$seeds" =~ ^[1-9][0-9]*$ && "$first" =~ ^[0-9]+$ ]]
mkdir -p "$(dirname "$out")"
mkdir "$out"
out=$(cd "$out" && pwd)
export TOKENWORLD_SIM_STAGE=${TOKENWORLD_SIM_STAGE:-/tmp/tokenworld-robotwin-paired-v1}
export ROBOTWIN_SIM_RUNTIME=${ROBOTWIN_SIM_RUNTIME:-"$(dirname "$(dirname "$release")")/zgy/robotwin_closedloop_20261021"}
echo "Output: $out (cross-segment cache mode: $mode)"
echo "Preparing RoboTwin on $(hostname) using $ROBOTWIN_SIM_RUNTIME"
export TOKENWORLD_RENDER_LIB="${TMPDIR:-/tmp}/tokenworld-render-$UID-17642827"
bash "$helper/prepare_renderer.sh" 2>&1 | tee "$out/renderer.log"
bash "$helper/prepare.sh" 2>&1 | tee "$out/prepare.log" || {
    status=${PIPESTATUS[0]}
    echo "RoboTwin preparation failed; see $out/prepare.log" >&2
    exit "$((status == 0 ? 1 : status))"
}
echo "Preparation complete. Launching $cards workers."
pids=()
for ((rank=0;rank<cards;rank++)); do
    bash "$release/scripts/run_python.sh" "$helper/robotwin_wm_pair.py" --release "$release" \
        --sim-root "$TOKENWORLD_SIM_STAGE/sim" --output "$out" --rank "$rank" --world-size "$cards" \
        --cache-mode "$mode" --device "cuda:$rank" --seed-start "$first" --seeds "$seeds" >"$out/rank$rank.log" 2>&1 &
    pids+=("$!")
    echo "Started rank $rank (pid $!) - $out/rank$rank.log"
    # Stagger checkpoint loads to reduce shared-storage and host RAM pressure.
    if ((rank+1<cards)); then sleep 8; fi
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if ((status)); then echo "Incomplete run. Inspect $out/rank*.log" >&2; exit 1; fi
python3 "$helper/summarize_pairs.py" --output "$out" --seed-start "$first" --seeds "$seeds"
