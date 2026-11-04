# All-task DeepStack adapter pipeline (PPU)

The pipeline exports cam_high features on eight PPUs, trains the adapter on one PPU, evaluates the selected checkpoint, and then deletes only this run's temporary feature shards. Existing S-VAE, World Model, RGB decoder and policy weights are frozen/unmodified. This uses the shared release-relative `venv` and PPU SDK.

## Submit one job

Choose **one node with eight PPUs** and shared storage. Paste this entire command in the training platform, including `bash -lc` (the platform's default `/bin/sh` does not support `set -o pipefail`):

```bash
bash -lc 'cd /mnt/world_foundational_model/chuyao/TokenWorld-release && bash scripts/reproduce_deepstack_adapter.sh --gpus 8 --output outputs/deepstack_adapter_all50_20261031'
```

The job should run once on the node; the script creates eight export workers itself. It is not an eight-node launcher. Once export workers exit, the same job trains on local PPU 0. The other seven PPUs remain reserved by the platform until the job finishes. Distributed rank/rendezvous variables are cleared at launcher boundaries so the one-process stages cannot accidentally join the export process group. Device visibility and SDK selection are preserved.

The release checkpoint in this output completed 33,720 updates. To train a new run with a different source revision, choose a new output directory.

## Data, storage and training

- All 50 tasks, both success/failure sources, official train/validation split, cam_high only.
- 4,499 training episodes, eight evenly spaced frames each: **35,992 training frames**.
- 500 validation episodes, four frames each: **2,000 validation frames**.
- No training/validation episode overlap. Success-outcome policy prompts are used consistently, including on failure-source images.
- Estimated temporary feature storage **79.1 GiB**, plus serialization overhead. Preflight requires 15% headroom plus 20 GiB for outputs/working space. This is sampled coverage of every episode, not export of every 50-Hz video frame.
- Adapter: unchanged 11,626,016-parameter pilot architecture. Existing latent and frozen merger predict the three post-merger DeepStack streams. No action distillation is included.
- Batch 32, AdamW, learning rate 3e-4 cosine to 3e-5, weight decay 0.01, gradient clipping 1, bf16, seed 3047. Default **30 epochs / 33,720 updates** (drop incomplete final minibatch; shuffle each epoch).
- Feature validation every 1,000 steps; action monitoring every 2,000 steps on a fixed 100-frame panel (one validation initial frame per task/outcome). Step 1 and the final step are also evaluated.
- The best checkpoint is selected by full validation feature loss. Final action evaluation runs on all 2,000 validation frames, reporting each task separately and comparing merger-only, adapter and true-DeepStack oracle against the historical RGB helper, which uses the post-normalization readout. For native policy agreement use the paired evaluation below. These are validation teacher-agreement metrics, not an independent test or simulator success rate.
- Training reads feature files through mmap and a bounded cache, with four CPU data-loader workers. It does not load the full feature dataset into Python tensors at once. Training-set normalization moments are combined from the export shards.

## Recovery and cleanup

Re-submit the **identical command** to reuse completed export files and resume the latest saved adapter/optimizer state. Source code, asset identities, data selection and training schedule must match the saved run contract. Use a fresh output directory when changing them. Exact epoch permutations continue at the saved optimizer step; unsaved work since the most recent checkpoint is repeated.

The stages are ordered: prepare -> export -> verify/aggregate -> train -> full action evaluation -> checkpoint integrity checks -> cleanup. A failed stage stops the shell script. No cleanup runs on training/evaluation failure.

Cleanup deletes only `features/*.pt` filenames explicitly recorded in this run's manifest. It requires a matching ownership marker, successful training/evaluation, matching checkpoint SHA-256 values, readable finite saved model tensors, expected feature file sizes, and no symlinks/unexpected files. It retains the tiny ownership marker and writes `cleanup_actions.jsonl` and `cleanup.json`. Neither the original S-VAE latent cache nor pilot features are deleted.

To retry interrupted cleanup alone, use the same arguments with `--stage cleanup`. Add `--keep-features` to retain the generated features after successful training. For a different deployment, override `IRASIM_ROOT`, `CACHE_ROOT`, `RAW_DATA_ROOT`, `CODEC_CHECKPOINT`, or `POLICY_BUNDLE` in the launcher environment.

Important retained files:

- `adapter_best.pt`, `adapter_last.pt` (the latter contains optimizer and resume metadata).
- `run.json`, `manifest.json`, source snapshots, `target_stats.pt`, `export_complete.json`.
- `feature_evaluation.json`, `action_monitor.json`, `action_evaluation.json`.
- `training_complete.json`, per-stage JSONL logs, cleanup records.

`export_complete.json` is the permanent inventory of exported files; after successful cleanup those feature files intentionally no longer exist. Re-running a completed, cleaned run does not recreate them.

## Local validation and closed-loop status

A two-PPU smoke run on ali-dev completed sharded extraction, export verification, training with two CPU workers, checkpoint selection, action monitoring, final action evaluation, and automatic deletion of its four feature files. Cleanup safety tests and deterministic resumed-batch-order tests passed; existing gripper-conversion tests also passed. The completed full run used eight export workers.

The original three-task pilot adapter was also tested in six full learned-WM closed-loop trajectories (1,301 frames, 28 policy queries). All saved policy queries replayed exactly, outputs were finite, and frame zero matched the cached GT latent. The current adapter **did not restore reliable closed-loop task behavior**: cabinet remained closed, and bottle/switch trajectories stalled or deviated. This is a different result from the 75.5% improvement in teacher-forced action agreement. Full-data training is an experiment, not a guarantee of closed-loop recovery.

Results: `outputs/adapter_closed_loop_20261031/`; comparison images/videos: its `visuals/` directory. Production default remains merger-only. To opt into the adapter, add `--adapter-checkpoint PATH` to `scripts/evaluate_closed_loop.sh` with the normal WM/codec/decoder/policy/manifest arguments. Use a fresh output root to separate interfaces and checkpoints.

## Native policy readout for the release QwenOFT bundle

QwenOFT trains its action head on the last decoder block output. In the pinned environment, the native RGB `predict_action` reads this tensor before the final text normalization. Closed-loop evaluation with `--adapter-checkpoint` defaults to `--adapter-readout native_pre_norm`. Use `--adapter-readout legacy_post_norm` to replay historical adapter trajectories. Use separate output directories for each readout.

The eight-PPU run selected weights by visual feature loss. Its historical action monitor, and the earlier 0.99236 cosine, compare outputs of the post-normalization helper. On the same 2,000 validation frames with the actual RGB policy as reference, the previous adapter readout scored 0.90671, the native readout scored 0.99463, and the RGB bridge scored 0.98401. All three use ground-truth observation latents. Each of the 500 sampled real-feature injection checks reproduced the saved native action chunk exactly.

With the saved RGB bridge validation reference in the run directory, reproduce this comparison from the release root:

```bash
bash scripts/run_python.sh scripts/evaluate_adapter_readout.py \
  --run-dir outputs/deepstack_adapter_all50_20261031 \
  --output outputs/adapter_readout_native_eval
```
