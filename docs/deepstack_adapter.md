# DeepStack adapter pilot

This is an exploratory **GT-latent representation** experiment, not a replacement for the release's default merger-only policy interface. It does not update the S-VAE, World Model, RGB decoder, or policy. It does not yet measure closed-loop success.

## Hypothesis and model

The existing cam_high S-VAE posterior mean `[108,16]` may retain enough information to predict the policy's three missing Qwen3-VL DeepStack streams. The frozen S-VAE decoder supplies the existing merger `[108,2560]` unchanged. A trainable adapter consumes that decoded merger and the latent, projects to width 512, applies two eight-head Transformer encoder blocks with learned spatial positions, and predicts three `[108,2560]` streams. Targets are the **post-merger** outputs from vision blocks 5, 11 and 17. Those streams enter the first three Qwen text blocks.

The first pilot uses feature supervision only: per-channel, train-set-standardized MSE, averaged equally over the three streams, plus 0.1 cosine distance on reconstructed features. Target means and standard deviations use training frames only. No action distillation is performed in this run. AdamW uses learning rate 3e-4, cosine decay to 3e-5, weight decay 0.01, batch 16, 2,000 updates, bf16 autocast and gradient norm clipping at 1. Seed is 3047.

## Data and validation

Tasks: `put_object_cabinet`, `shake_bottle_horizontally`, `turn_switch`; cam_high only. Select 12 training episodes and two validation episodes per task/outcome group from the existing cache's official split, with deterministic shuffling. Use eight evenly spaced training frames and four validation frames per episode: 72 episodes / 576 frames for training, 12 episodes / 48 frames for validation. Train and validation episode identifiers must be disjoint. All frames use the existing unnormalized posterior-mean latent, not a WM prediction. Prompts request the success outcome, including failure-source observations, matching the diagnostic closed-loop setup.

For every validation frame, require native RGB actions and injected **true merger + true DeepStack** actions to agree within max absolute normalized-action error 1e-5. A failed check aborts the experiment. Compare:

- Native RGB (teacher).
- True merger only (missing-stream diagnostic).
- Frozen S-VAE decoded merger only (current token interface).
- Frozen S-VAE decoded merger + true DeepStack (oracle diagnostic).
- Frozen S-VAE decoded merger + training-set mean DeepStack (constant-feature control).
- Frozen S-VAE decoded merger + adapter-predicted DeepStack (candidate).

The primary action metric is physical joint MAE relative to the native RGB teacher over its 50-action chunk; also report binary-gripper agreement. Recorded-action errors are secondary diagnostics; failure-trajectory recorded actions are not a success target. The best checkpoint is selected on validation feature loss, so its action results are validation results, **not an independent test**. Episode/task-level results must be inspected rather than interpreting 48 correlated frames as 48 independent episodes.

## Run on ali-dev / shared PPU storage

```bash
cd /mnt/world_foundational_model/chuyao/TokenWorld-release
bash scripts/train_deepstack_adapter.sh \
  --output outputs/deepstack_adapter_pilot_20261031 \
  --device cuda:0
```

Use a new output directory for every run; the script refuses to overwrite an existing run. The launcher uses the release-relative `venv` and PPU SDK via `scripts/run_python.sh`. Override `IRASIM_ROOT` or `CACHE_ROOT`, `RAW_DATA_ROOT`, `CODEC_CHECKPOINT`, and `POLICY_BUNDLE` when those assets live elsewhere. No installation is needed on the present ali-dev environment.

The script automatically extracts targets, trains, selects a checkpoint, and evaluates policy actions. Artifacts are `config.json`, `manifest.json`, `features/`, `metrics.jsonl`, `status.json`, `adapter_best.pt`, `adapter_last.pt`, `feature_evaluation.json`, and `action_evaluation.json`. Feature targets are small pilot artifacts, not a full-dataset export. The last checkpoint includes optimizer state for inspection; automatic resume is not implemented.

A successful feature reconstruction alone does not establish policy compatibility. If the oracle helps but the adapter does not, inspect representational capacity, loss weighting and data coverage before concluding that the latent necessarily lost information. If the adapter improves held-out actions, next test WM-predicted latents and full closed-loop rollouts; distribution shift may still be material. A small three-task pilot cannot establish all-task reproduction.
