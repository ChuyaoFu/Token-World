#!/usr/bin/env bash
# Prepared for manual scheduler submission. Never submits a job itself.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
run=${RUN_ROOT:-outputs/decoder_improvement_20261102}
out=${1:-outputs/decoder_shape_full_8ppu_$(date -u +%Y%m%dT%H%M%SZ)}
args=(--architecture "${ARCHITECTURE:-baseline}" --baseline "$root/outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt"
 --output "$out" --validation-probes "$run/validation_extended_guard.pt"
 --cache "$root/../IRASim/cache/robotwin_all50_symmetric_qwen_svae_dim16_50hz"
 --data-root "$root/../IRASim/robotdata/robotwin-success-fail-lerobot"
 --codec "$root/outputs/svae_d16_3views/best.pt"
 --supplement-probes "$run/probes_sim_train.pt" --supplement-fraction 0.25
 --steps "${STEPS:-5000}" --batch-size "${BATCH_SIZE_PER_RANK:-2}" --frames-per-episode 2
 --workers 2 --lr 1e-5 --raw-fraction 0.5 --lpips-weight 0.05 --edge-weight 0.1
 --l1-weight 0.1 --boundary-weight 0.1 --checkpoint-every 1000)
command=(bash scripts/run_python.sh -m torch.distributed.run --standalone --nproc_per_node=8
 --master_port "${MASTER_PORT:-29615}" scripts/decoder_improvement/train.py "${args[@]}")
printf '%q ' "${command[@]}"; printf '\n'
if [[ ${DRY_RUN:-0} == 1 ]]; then exit 0; fi
[[ -f "$run/validation_extended_guard.pt" && -f "$run/probes_sim_train.pt" ]] || { echo 'Run prepare_sim.py first'; exit 2; }
[[ ! -e "$out" ]] || { echo 'Choose a fresh output directory'; exit 2; }
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false TORCH_HOME="$root/outputs/pretrained/torch"
exec "${command[@]}"
