#!/usr/bin/env bash
# Run the release adapter against the matched RoboTwin reset panel.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
helper="$root/scripts/robotwin_sim"
cards=${1:-8}
count=${2:-25}
first=${3:-100000}
mode=${ROBOTWIN_CACHE_MODE:-gru}
case "$mode" in gru|full) ;; *) echo "Expected ROBOTWIN_CACHE_MODE=gru or full" >&2; exit 2;; esac
out=${4:-"$root/outputs/robotwin_${mode}_cache_$(date -u +%Y%m%dT%H%M%SZ)"}
reference=${ROBOTWIN_REFERENCE:-"$root/outputs/robotwin_wm_pair_20261101T080111Z"}
panel=${ROBOTWIN_PANEL:-"$reference"}
if [[ -z "${ROBOTWIN_PANEL:-}" && -d "$reference/recovery_ali_dev/complete_panel" ]]; then
    panel="$reference/recovery_ali_dev/complete_panel"
fi
[[ "$cards" =~ ^[1-9][0-9]*$ && "$count" =~ ^[1-9][0-9]*$ && "$first" =~ ^[0-9]+$ ]]
reference=$(cd "$reference" && pwd)
panel=$(cd "$panel" && pwd)
read -ra tasks <<< "${ROBOTWIN_TASKS:-put_object_cabinet shake_bottle_horizontally turn_switch}"
mapfile -t seeds < <(seq "$first" "$((first+count-1))")
mkdir -p "$(dirname "$out")"
mkdir "$out"
out=$(cd "$out" && pwd)
export TOKENWORLD_SIM_STAGE=${TOKENWORLD_SIM_STAGE:-/tmp/tokenworld-robotwin-paired-v1}
export ROBOTWIN_SIM_RUNTIME=${ROBOTWIN_SIM_RUNTIME:-"$(dirname "$(dirname "$root")")/zgy/robotwin_closedloop_20261021"}
export TOKENWORLD_RENDER_LIB=${TOKENWORLD_RENDER_LIB:-"${TMPDIR:-/tmp}/tokenworld-render-$UID-17642827"}
echo "Output: $out (cross-segment cache mode: $mode)"
bash "$helper/prepare_renderer.sh" 2>&1 | tee "$out/renderer.log"
bash "$helper/prepare.sh" 2>&1 | tee "$out/prepare.log"
pids=()
for ((rank=0; rank<cards; rank++)); do
    bash "$root/scripts/run_python.sh" "$helper/evaluate_full_cache.py" --release "$root" \
        --cache-mode "$mode" --reference "$reference" --panel "$panel" --output "$out" --sim-root "$TOKENWORLD_SIM_STAGE/sim" \
        --seeds "${seeds[@]}" --tasks "${tasks[@]}" --rank "$rank" --world-size "$cards" --device "cuda:$rank" \
        >"$out/rank$rank.log" 2>&1 &
    pids+=("$!")
    echo "Started rank $rank (PID $!); see $out/rank$rank.log"
    if ((rank+1<cards)); then sleep 5; fi
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if ((rc)); then echo "Incomplete evaluation. Inspect $out/rank*.log" >&2; exit 1; fi
python3 "$helper/summarize_full_cache.py" --cache-mode "$mode" --output "$out" --tasks "${tasks[@]}" --seeds "${seeds[@]}"
