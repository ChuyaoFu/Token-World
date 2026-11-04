#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
[[ $# -ge 2 ]] || { echo 'Usage: bash scripts/decoder_improvement/evaluate_candidate.sh CHECKPOINT OUTPUT_DIR [PROBES]'; exit 2; }
checkpoint=$1
out=$2
probes=${3:-outputs/decoder_improvement_20261102/broad_val/probes.pt}
[[ -f "$checkpoint" && -f "$probes" ]] || { echo 'Missing checkpoint or fixed probes'; exit 2; }
[[ ! -e "$out" ]] || { echo 'Choose a fresh output directory'; exit 2; }
mkdir -p "$out"
bash scripts/run_python.sh scripts/decoder_improvement/make_rois.py --probes "$probes" --output "$out/rois.json"
bash scripts/run_python.sh scripts/decoder_improvement/evaluate.py --probes "$probes" --decoder "$checkpoint" --output "$out" --device "${DEVICE:-cuda:0}" --roi-json "$out/rois.json"
