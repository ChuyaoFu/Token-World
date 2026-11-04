#!/usr/bin/env bash
# Run inside a user-submitted single-node eight-PPU allocation. Resumable.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
out=${1:-outputs/robotwin_closed_loop_vs_sim_all50x10}
cards=${NUM_PPUS:-8}
per_task=${PER_TASK:-10}
first=${SEED_START:-100000}
attempts=${MAX_ATTEMPTS:-100}
max_steps=${MAX_STEPS:-0}
[[ $cards =~ ^[1-8]$ ]] || { echo 'NUM_PPUS must be 1..8'; exit 2; }
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$root/outputs/pretrained/torch"
export TOKENWORLD_SIM_STAGE=${TOKENWORLD_SIM_STAGE:-/tmp/tokenworld-robotwin-paired-v1}
export ROBOTWIN_SIM_RUNTIME=${ROBOTWIN_SIM_RUNTIME:-"$(dirname "$(dirname "$root")")/zgy/robotwin_closedloop_20261021"}
export TOKENWORLD_RENDER_LIB=${TOKENWORLD_RENDER_LIB:-${TMPDIR:-/tmp}/tokenworld-render-$UID-17642827}
stage=${ALL_TASK_SIM_STAGE:-/tmp/tokenworld-robotwin-all50-v1/sim}
objects=${ROBOTWIN_ALL_OBJECTS:-$root/outputs/robotwin_sim_runtime/objects-all50-3dc3b798}
args=(--sim-root "$stage" --output "$out" --per-task "$per_task" --seed-start "$first"
      --max-attempts "$attempts" --max-steps "$max_steps")
tasks=()
if [[ -n ${ROBOTWIN_TASKS:-} ]]; then read -r -a tasks <<< "$ROBOTWIN_TASKS"; args+=(--tasks "${tasks[@]}"); fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'all50 x %s valid resets; %s workers; seed start %s; output %s\n' "$per_task" "$cards" "$first" "$out"
    printf 'bash scripts/run_python.sh scripts/closed_loop_video/run.py '; printf '%q ' "${args[@]}"; printf '\n'
    exit 0
fi
mkdir -p "$out"
exec 9>"$out/.run.lock"
flock -n 9 || { echo "Another launcher is using $out" >&2; exit 1; }
bash scripts/robotwin_sim/prepare_renderer.sh 2>&1 | tee "$out/renderer.log"
bash scripts/robotwin_sim/prepare.sh 2>&1 | tee "$out/prepare_base.log"
bash scripts/run_python.sh scripts/closed_loop_video/prepare_assets.py \
    --base "$TOKENWORLD_SIM_STAGE/sim" --stage "$stage" --cache "$objects" 2>&1 | tee "$out/prepare_all.log"
bash scripts/run_python.sh scripts/closed_loop_video/run.py "${args[@]}" --prepare 2>&1 | tee "$out/contract.log"
preflight=(--sim-root "$stage" --output "$out" --seed "$first")
if (( ${#tasks[@]} )); then preflight+=(--tasks "${tasks[@]}"); fi
bash scripts/run_python.sh scripts/closed_loop_video/preflight.py "${preflight[@]}" 2>&1 | tee "$out/preflight.log"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
for ((rank=0;rank<cards;rank++)); do
    bash scripts/run_python.sh scripts/closed_loop_video/run.py "${args[@]}" \
        --rank "$rank" --world-size "$cards" >"$out/rank$rank.log" 2>&1 &
    pids+=("$!")
    echo "Started rank $rank: $out/rank$rank.log"
    if ((rank+1<cards)); then sleep 8; fi
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if ((status)); then echo "Incomplete run; inspect $out/rank*.log and rerun the same command." >&2; exit 1; fi
bash scripts/run_python.sh scripts/closed_loop_video/run.py "${args[@]}" --summarize
echo "Complete: $out/summary.json; videos: $out/<task>/<seed>/comparison.mp4"
