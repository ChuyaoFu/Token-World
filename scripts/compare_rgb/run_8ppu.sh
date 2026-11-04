#!/usr/bin/env bash
# Run inside a manually submitted eight-PPU job; this script never submits a job.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
selection=${SELECTION:-all500}
out=${1:-outputs/robotwin_rgb_${selection}_current_25hz}
reuse_from=${REUSE_FROM-$root/outputs/robotwin_rgb_compare_20261102/pilot_current}
cards=${NUM_PPUS:-8}
distribution=${DISTRIBUTION_METRICS:-1}
ctrl_python=${CTRL_PYTHON:-/mnt/world_foundational_model/chuyao/venvs/tokenworld-ppu/bin/python}
[[ $cards =~ ^[1-8]$ ]] || { echo 'NUM_PPUS must be 1..8'; exit 2; }
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false TORCH_HOME="$root/outputs/pretrained/torch"
prepare=(bash scripts/run_python.sh scripts/compare_rgb/prepare_panel.py --selection "$selection" --output "$out/panel.json")
reuse=(bash scripts/run_python.sh scripts/compare_rgb/reuse_artifacts.py --source-root "$reuse_from" --source-panel "$reuse_from/panel.json" --panel "$out/panel.json" --output "$out")
token=(bash scripts/run_python.sh -m torch.distributed.run --standalone --nproc_per_node="$cards" --master_port 29641 scripts/compare_rgb/replay_tokenworld.py --panel "$out/panel.json" --output "$out")
ctrl=(bash scripts/run_python.sh -m torch.distributed.run --standalone --nproc_per_node="$cards" --master_port 29642 scripts/compare_rgb/replay_ctrlworld.py --panel "$out/panel.json" --output "$out" --grid 25)
score=(bash scripts/run_python.sh -m torch.distributed.run --standalone --nproc_per_node="$cards" --master_port 29643 scripts/compare_rgb/score.py --panel "$out/panel.json" --artifacts "$out" --output "$out/scores" --grid 25)
summarize=(bash scripts/run_python.sh scripts/compare_rgb/summarize.py --panel "$out/panel.json" --scores "$out/scores" --output "$out/report" --grid 25)
if [[ ${DRY_RUN:-0} == 1 ]]; then
 printf '%q ' "${prepare[@]}"; printf '\n'; if [[ -n "$reuse_from" && "$reuse_from" != "$out" && -f "$reuse_from/panel.json" ]]; then printf '%q ' "${reuse[@]}"; printf '\n'; fi; printf '%q ' "${token[@]}"; printf '\n'; printf 'PYTHON_BIN=%q ' "$ctrl_python"; printf '%q ' "${ctrl[@]}"; printf '\n'; printf '%q ' "${score[@]}"; printf '\n'; printf '%q ' "${summarize[@]}"; printf '\n'; if [[ $distribution == 1 ]]; then printf 'NUM_PPUS=%q bash scripts/compare_rgb/score_distribution.sh %q\n' "$cards" "$out"; fi; exit 0
fi
mkdir -p "$out"
if [[ $distribution == 1 ]]; then bash scripts/run_python.sh scripts/compare_rgb/setup_distribution.py 2>&1 | tee "$out/distribution_setup.log"; fi
"${prepare[@]}" 2>&1 | tee "$out/prepare.log"
if [[ -n "$reuse_from" && -f "$reuse_from/panel.json" && "$(cd "$reuse_from" && pwd -P)" != "$(cd "$out" && pwd -P)" ]]; then
 "${reuse[@]}" 2>&1 | tee "$out/reuse.log"
fi
"${token[@]}" 2>&1 | tee "$out/tokenworld.log"
PYTHON_BIN="$ctrl_python" "${ctrl[@]}" 2>&1 | tee "$out/ctrlworld.log"
"${score[@]}" 2>&1 | tee "$out/scores.log"
"${summarize[@]}" 2>&1 | tee "$out/summary.log"

if [[ $distribution == 1 ]]; then NUM_PPUS="$cards" bash scripts/compare_rgb/score_distribution.sh "$out" 2>&1 | tee "$out/distribution.log"; fi
