#!/usr/bin/env bash
# Score Token-World on the balanced 100-episode RoboTwin Table I panel.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
helper="$root/scripts/table1_robotwin"
assets=${IRASIM_ROOT:-"$root/../IRASim"}
bundle=${POLICY_BUNDLE:-"$assets/results/action_expert/tokenworld_qwenoft_success_cam_high_teacher"}
cards=${1:-2}
mode=${CACHE_MODE:-gru}
interface=${ACTION_INTERFACE:-adapter}
out=${2:-"$root/outputs/table1_robotwin_${mode}_${interface}_$(date -u +%Y%m%dT%H%M%SZ)"}
[[ $cards =~ ^[1-9][0-9]*$ ]] && ((cards <= 100)) || { echo "Expected 1–100 workers"; exit 2; }
case "$interface" in adapter|merger_only) ;; *) echo "ACTION_INTERFACE must be adapter or merger_only"; exit 2 ;; esac
case "$mode" in gru|reset) ;; *) echo "CACHE_MODE must be gru or reset"; exit 2 ;; esac
mkdir -p "$(dirname "$out")"; mkdir "$out"
out=$(cd "$out" && pwd)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} TOKENIZERS_PARALLELISM=false PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
echo "Table I RoboTwin | cache_mode=$mode | action_interface=$interface | output=$out"
bash "$root/scripts/run_python.sh" -m torch.distributed.run --standalone --nproc_per_node="$cards" "$helper/evaluate.py" \
    --cache "${LATENT_CACHE:-$assets/cache/robotwin_all50_symmetric_qwen_svae_dim16_50hz}" \
    --checkpoint "${WM_CHECKPOINT:-$root/outputs/world_model_svae_d16_cam_high_100k/checkpoint_step_0100000.pt}" \
    --codec-checkpoint "${CODEC_CHECKPOINT:-$root/outputs/svae_d16_3views/best.pt}" \
    --reference-report "${REFERENCE_REPORT:-$helper/panel.json}" \
    --data-root "${RAW_DATA_ROOT:-$assets/robotdata/robotwin-success-fail-lerobot}" \
    --qwen-model "${QWEN_MODEL:-$assets/pretrained_models/Qwen3-VL-4B-Instruct}" \
    --action-config "$bundle/model/config.full.yaml" --action-checkpoint "$bundle/model/checkpoints/steps_50000_pytorch_model.pt" \
    --action-interface "$interface" --adapter-checkpoint "${ADAPTER_CHECKPOINT:-$root/outputs/deepstack_adapter_all50_20261031/adapter_best.pt}" \
    --cache-mode "$mode" --output "$out" 2>&1 | tee "$out/run.log"
python3 "$helper/summarize.py" "$out"
