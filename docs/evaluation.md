# Evaluation protocols

## Open-loop

```bash
python -m tokenworld.evaluation.open_loop \
  --cache /path/to/svae_cache \
  --world-model-checkpoint /path/to/world_model.pt \
  --codec-checkpoint /path/to/codec.pt \
  --qwen-model /path/to/Qwen3-VL-4B-Instruct \
  --raw-data-root /path/to/robotwin-success-fail-lerobot \
  --policy-bundle /path/to/qwenoft_bundle \
  --views 0 --reference raw_teacher --sampling-steps 16 \
  --initial-context-frames 1 --split val --output outputs/open_loop.json
```

The policy bundle is optional for feature-only evaluation and required for
action consistency. Its current adapter is OFT-specific; PIv3 and GR00T are
not silently routed through OFT. The bundle's own model config may contain
machine paths and must be reviewed when checkpoints are selected.

The workspace evaluator defaults to one ground-truth initialization frame;
four-frame initialization remains an explicit option. Keep this setting,
selected episodes, camera, reference mode and query interval in each result.
Compare raw frozen-Qwen GT features for `raw_teacher`. `codec_gt` compares
against reconstructed GT features and is a different experiment.

The release preserves micro aggregates and adds `aggregate_macro`: average
episode metrics within each task, then give each task equal weight, matching
the paper's description. This is not a claim that existing micro-averaged
numbers equal the published table. Shards can be aggregated with:

```bash
python -m tokenworld.evaluation.aggregate \
  --input outputs/shard0.json outputs/shard1.json --output outputs/macro.json
```

Duplicate episodes, mismatched protocols and incomplete metric coverage are
rejected. No plotting code is included.

## Meaning of the action metric

The current QwenOFT adapter injects **final merger features only**. Both the
predicted-feature and reference-feature branches omit Qwen3-VL DeepStack visual
injections. Therefore this metric measures action consistency through that
merger-only interface; it is not agreement with the full native image-policy
path. `raw_teacher` describes the feature reference, not a native full-policy
action reference. The output now records `action_feature_interface` and
`action_reference` explicitly, plus the raw-data and Qwen paths.

A controlled audit found exact agreement with native inference when DeepStack
was disabled, but substantial differences from native inference with DeepStack
active. Changing the action reference can even reverse checkpoint rankings on
some trajectories. Keep this diagnostic separate from full-policy fidelity and
simulator task success. Do not supply ground-truth future DeepStack features to
the predicted branch: that would introduce future visual information.

The action metric aggregates the policy full 50-action chunk, not only its first
executed action. Report first-action errors separately when diagnosing a failure.

## Long horizon and timing

The paper reports chunks 0, 5, 10 and 15, with 16 actions per chunk. The exact
paper episode subset and chunk initialization/reset rules must be recovered
before declaring these curves reproducible. The release now reports `chunks` per episode (default `--chunk-size 16`),
including exact predicted-frame boundaries and whether a final chunk is partial.
These are contiguous evaluation windows without GT resets; the original paper
window protocol still needs comparison. No plots are generated.

TokenWorld times dynamics plus conversion to policy-facing features and
excludes the reference/policy scoring branches. RGB baselines must include
RGB decoding and Qwen re-encoding. Report per generated observation, batch 1,
warmup policy, device and sampling steps. The paper used PPU-ZW810E, 16 steps
for TokenWorld and 50 for RGB baselines. A CUDA timing measurement is a new
measurement, not a reproduction of those hardware-specific values.

## Closed loop

`tokenworld.evaluation.closed_loop` is the recovered RoboTwin OFT replay runner. It retains
the original action-order conversion, normalization, policy-query interval
and predicted-state feedback. Its defaults are inherited workspace settings,
not yet certified paper settings. See [integration status](integrations.md) before use.

Manual success labels must identify task, policy, seed/initial state, world
model, trial and the video reviewed. Preserve reference-environment outcomes
for the exact matching initial states. The combined RoboTwin/Franka Pearson
coefficients in Figure 6 are not RoboTwin-only acceptance targets. Reference
simulation, three-head policy parity and manual-label records remain pending.

## RGB replay and learned-world-model closed loop

The following entrypoints use the checkout-relative PPU environment:

- `scripts/replay_world_model.sh`: recorded-action open-loop replay, ported from `replay_tokenworld_svae_latents.py` with the original normalization, action alignment and random-seed schedule.
- `scripts/evaluate_closed_loop.sh`: original QwenOFT policy interacting with predicted visual tokens and predicted proprioception. The merger-only policy interface is preserved. RGB decoding is for visualization; it is not fed back to the policy.
- `scripts/render_replay.sh`: matched raw RGB / GT-latent reconstruction / open-loop replay / learned-world-model closed-loop comparison videos, contact sheets, and RGB reconstruction metrics. Requires both replay jobs to finish first.

The release defaults to `--cache-mode gru` in both recorded-action and closed-loop replay. KV accumulates within a segment. Across segments, the GRU hidden state preceding the overlapping frame seeds the next segment, which rebuilds KV from that frame. Use `--cache-mode reset` to reproduce the archived segment-reset baseline or `--cache-mode full` to carry the complete temporal cache. Episode metadata records the selected mode.

This is a separate protocol from `evaluate_world_model.sh`'s one-frame-at-a-time feature/action fidelity. The original replay contract uses 15 generated frames per block and one initial observed frame. Closed-loop re-queries the policy every 50 actions, always with a success-conditioned prompt (including initial states selected from failure trajectories). All later visual/proprioceptive inputs are predictions. Explicitly record sampling steps; the example uses 16, while recovered replay entrypoints retain their original default of 64. Videos use the source cache rate of 50 Hz; older 25-fps exports are slow playback.

Prepare a manifest (use `--all-validation` for all 500 validation episodes):

```bash
cd /path/to/TokenWorld-release
CACHE=/path/to/robotwin_all50_symmetric_qwen_svae_dim16_50hz
RAW=/path/to/robotwin-success-fail-lerobot
POLICY=/path/to/tokenworld_qwenoft_success_cam_high_teacher
OUT=outputs/replay_evaluation
WM=outputs/world_model_svae_d16_cam_high_100k/checkpoint_step_0100000.pt
CODEC=outputs/svae_d16_3views/best.pt
DECODER=outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt

bash scripts/run_python.sh -m tokenworld.data.select_replay \
  --index "$CACHE/index.json" --output "$OUT/manifest.json" --episodes 10 --seed 3047

bash scripts/replay_world_model.sh \
  --manifest "$OUT/manifest.json" --cache "$CACHE" --checkpoint "$WM" \
  --codec-checkpoint "$CODEC" --output-dir "$OUT/open_loop" \
  --execution-horizon 15 --num-denoise-steps 16 --device cuda:0

bash scripts/evaluate_closed_loop.sh \
  --manifest "$OUT/manifest.json" --cache "$CACHE" --world-model-checkpoint "$WM" \
  --codec-checkpoint "$CODEC" --decoder-checkpoint "$DECODER" --policy-bundle "$POLICY" \
  --output-root "$OUT/closed_loop" --execution-horizon 15 --query-every 50 --gripper-mode binary \
  --num-denoise-steps 16 --fps 50 --device cuda:0

bash scripts/render_replay.sh \
  --manifest "$OUT/manifest.json" --cache "$CACHE" --raw-data-root "$RAW" \
  --codec-checkpoint "$CODEC" --decoder-checkpoint "$DECODER" \
  --open-loop-root "$OUT/open_loop" --closed-loop-root "$OUT/closed_loop" \
  --output-root "$OUT/comparison" --device cuda:0
```

The selector balances outcomes, not tasks. Use an explicitly prepared matched manifest when reporting per-task results. The 2026-10-31 diagnostic manifest contains the first validation success/failure episode of each of `put_object_cabinet`, `shake_bottle_horizontally`, and `turn_switch`, with their full recorded lengths.

Renderer checks finite latents/actions, exact observed frame-zero initialization, frame counts, and policy-query counts. Open-loop metrics exclude frame zero. Closed-loop outputs are qualitative: actions differ from recorded actions, so raw-video similarity is not a success metric. No simulator success rate, reward-model score, or paper correlation is inferred. Formal closed-loop success evaluation still requires reviewed success labels or a validated scorer plus matched RoboTwin reference-environment runs. Use fresh output directories when changing protocols; `--skip-existing` does not validate all provenance fields.

## Closed-loop action-conversion audit (2026-10-31)

The QwenOFT bundle trains the 12 arm joints with min/max normalization to [-1,1], but its two grippers use binary targets in [0,1] with threshold 0.49. The recovered original runner incorrectly applied inverse min/max to all 14 dimensions. `evaluate_closed_loop.sh` now defaults to `--gripper-mode binary`; use `--gripper-mode legacy_minmax` only to reproduce historical outputs, including `outputs/replay_closed_loop_20261031`. The mode is recorded in new metadata. Joint ordering/scaling and the merger-only policy interface are unchanged. Correcting this conversion does not certify closed-loop task success; the missing DeepStack path and long action horizon are separate diagnostic concerns.

Policy-interface diagnostics are available as `bash scripts/run_python.sh scripts/diagnose_closed_loop_policy.py --replay-run PATH --raw-data-root PATH --output PATH --device cuda:0`. This audits native RGB, raw merger-only, codec-GT merger-only, and stored predicted-latent inputs. GT branches after frame zero are teacher-forced diagnostics, never closed-loop success estimates. Its inference-mode audit is not promised bit-identical to the historical grad-enabled runtime; `initial_query_exact_replay.json` separately verifies the production initial queries.

For isolated rollout interventions, replace the `evaluate_closed_loop.sh` invocation with `bash scripts/run_python.sh scripts/diagnose_closed_loop_rollout.py --diagnostic-variant gripper` (same remaining arguments) or `--diagnostic-variant rgb_gripper --raw-data-root PATH`. Use fresh output directories. RGB mode uses raw RGB only for frame zero and model-predicted RGB thereafter; it is an exploratory diagnostic, not the TokenWorld token-interface recipe. Query intervals can be overridden with `--query-every`. Neither variant automatically scores success.


## Optional DeepStack adapter

Add `--adapter-checkpoint PATH` to `scripts/evaluate_closed_loop.sh` to predict three DeepStack streams from the current raw (unnormalized) S-VAE latent and its frozen decoded merger. The policy receives those streams directly through the Qwen text backbone; RGB decoding is used only to visualize the rollout. The default remains merger-only. Metadata records the interface, absolute adapter checkpoint path and SHA-256; `--skip-existing` rejects a different adapter path/content.

The initial frame is the cached GT latent; subsequent observations and proprioception come from the WM. Use a fresh output root when comparing interfaces. Do not interpret feature validation or native-policy action agreement as simulator task success. The 2026-10-31 pilot's six full closed-loop trajectories remain qualitatively unsuccessful despite improved offline action agreement; see `outputs/adapter_closed_loop_20261031/REPORT.md`.
