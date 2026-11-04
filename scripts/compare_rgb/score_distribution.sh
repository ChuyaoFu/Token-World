#!/usr/bin/env bash
# Post-process existing RGB; no world model or RGB decoder is run here.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
out=${1:-outputs/robotwin_rgb_all500_current_25hz}
cards=${NUM_PPUS:-8}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME="$root/outputs/pretrained/torch"
bash scripts/run_python.sh scripts/compare_rgb/setup_distribution.py
bash scripts/run_python.sh -m torch.distributed.run --standalone --nproc_per_node="$cards" --master_port 29644 \
 scripts/compare_rgb/extract_distribution.py --panel "$out/panel.json" --artifacts "$out" --output "$out/distribution_features"
bash scripts/run_python.sh scripts/compare_rgb/aggregate_distribution.py \
 --panel "$out/panel.json" --features "$out/distribution_features" --output "$out/distribution_report"
