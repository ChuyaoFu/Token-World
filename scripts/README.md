# Reproduction entrypoints

Install once from the repository root with `bash scripts/setup_env.sh`. This
creates `venv/` using Python 3.12 and the PPU SDK's system torch/torchvision,
then installs project dependencies with `configs/environment/ppu.txt` constraints.
`PPU_PYTHON` selects the base Python. `PPU_SDK_SETUP` overrides the SDK setup path.
See the [root README](../README.md#installation) for shared-node requirements.

The training scripts locate `venv/bin/python` relative to this checkout and load
the PPU SDK automatically. They can be called from any working directory; user
supplied relative paths are interpreted relative to the repository root.
`--help` and `--dry-run` work without installing dependencies or loading the SDK.

Use the same environment for diagnostics, cache export and tests:

```bash
bash scripts/run_python.sh -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
bash scripts/run_python.sh -m tokenworld.data.export_svae --help
```

## Workflow

| Script | Purpose |
| --- | --- |
| `setup_env.sh` | Install the PPU environment into the checkout's `venv/` |
| `train_svae.sh` | Train the S-VAE; use `--views all` for the three-camera reproduction |
| `reproduce_world_model.sh` | Export, finalize, check and train; supports separate stages |
| `check_world_model_cache.py` | Check finalized cache coverage and sampled training/validation tensors |
| `train_rgb_decoder.sh` | Train the cam_high RGB decoder; preflight, smoke test and resume supported |
| `train_world_model.sh` | Lower-level world-model launcher with native argument forwarding |
| `evaluate_robotwin.sh` | Main GRU adapter evaluation on verified RoboTwin resets |
| `evaluate_robotwin_paired.sh` | RoboTwin simulator closed loop with intra-segment KV and cross-segment GRU |
| `evaluate_world_model.sh` | Open-loop feature/action fidelity on matched validation episodes |
| `validate_world_model.py` | Fixed-seed training-objective validation over all configured windows |
| `compare_world_model_evaluations.py` | Validate paired reports and compare task macro metrics |
| `run_python.sh` | Run other Python commands with the same PPU SDK and venv |

All external model/data/checkpoint locations are supplied as arguments. The
workflow does not require the original research workspace or hard-coded user
paths. Obtain the Qwen model, RoboTwin data and an S-VAE checkpoint separately,
or produce the checkpoint with Step 1. Installation and these scripts target
PPU nodes with compatible Python/SDK system packages.

## 1. S-VAE training

```bash
bash scripts/train_svae.sh \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --data-root /path/to/robotwin-success-fail-lerobot \
  --views all --gpus 8 \
  --output outputs/svae_d16_3views
```

Training and validation use **cam_high only** by default. Add `--views all` to
train and validate on cam_high plus both wrist cameras; the default output then
becomes `outputs/svae_d16_3views`, and the 8-process epoch has 46,143 steps.
Use a separate output directory and do not resume from the single-camera run. The script launches
`tokenworld.codec.train` with torchrun, using d=16, a 9x12 grid, one epoch,
8 images per process, learning rate 1e-4 and seed 3047. Training uses all frames;
validation samples up to eight frames per episode. These are the current release
recipe defaults, not certification of the final paper training budget.

Add `--dry-run` to print the command, or `--plan-only` to let the trainer inspect
metadata and report actual samples/steps without loading the model or training.
The latter requires the Qwen config, dataset metadata and manifest to exist.
Other native trainer arguments override script defaults, for example
`--epochs 3 --batch-size 4`. When changing token dimension, set a distinct
`--output` directory too.

Outputs include `training_plan.json`, `best.pt`, `final.pt`, `last.pt`,
`epoch_XXXX.pt` and, after normal completion, `metrics.json`. During an interrupted
run, `resume.pt` stores the optimizer/scheduler state. The trainer auto-resumes
from that file in the same output directory. To resume explicitly:

```bash
bash scripts/train_svae.sh \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --data-root /path/to/robotwin-success-fail-lerobot \
  --gpus 8 --output outputs/svae_d16 \
  --resume outputs/svae_d16/resume.pt
```

Use the same run configuration for resume. `--no-auto-resume` disables automatic
recovery; use a new output directory for a fresh experiment.

## 2. World-model reproduction: export through training

After the three-camera S-VAE finishes, the complete next-stage command is:

```bash
bash scripts/reproduce_world_model.sh \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --data-root /path/to/robotwin-success-fail-lerobot \
  --codec-checkpoint outputs/svae_d16_3views/best.pt \
  --gpus 8
```

The default `--stage all` executes these dependent steps in order:

1. Export all three camera grids at 50 Hz, using posterior means and Qwen image batch 12.
2. Finalize `index.json`, `cache_spec.json` and training-only normalization statistics after all export workers exit successfully.
3. Check all indexed shard paths, exact episode assignments, and six sampled windows for model-compatible shapes and finite values.
4. Train the world model on cam_high/view 0 using the default 8-PPU, 100k-step recipe.

The default cache is `outputs/cache/robotwin_svae_d16_3views_50hz`; the training
output is `outputs/world_model_svae_d16_cam_high_100k`. Override these with
`--cache` and `--output`. Use `--config` and `--manifest` to select alternative
recipes/splits. The checker verifies file coverage and sampled tensors; it does
not deserialize every shard. The workflow stops when any stage fails.

Use `--stage export` with the same inputs to stop after export/finalization/checks.
If export is interrupted, rerun with identical inputs: existing shards are
reused, and the exporter's cache-plan consistency check rejects a different plan.
No `--overwrite` is passed by this wrapper. Use a new cache path for another codec.

For a cache already finalized with the intended codec:

```bash
# Optional standalone check, without allocating devices.
bash scripts/reproduce_world_model.sh --stage check \
  --cache /path/to/finalized_cache

# The training stage repeats the cache check before starting.
bash scripts/reproduce_world_model.sh --stage train \
  --cache /path/to/finalized_cache \
  --gpus 8 \
  --output outputs/world_model_svae_d16_cam_high_100k
```

`--stage train` needs only the cache, not Qwen weights/raw videos. It preserves
cache provenance rather than claiming a new export. World-model resume is explicit:

```bash
bash scripts/reproduce_world_model.sh --stage train \
  --cache /path/to/finalized_cache \
  --gpus 8 \
  --output outputs/world_model_svae_d16_cam_high_100k \
  --resume outputs/world_model_svae_d16_cam_high_100k/checkpoint_step_0010000.pt
```

Without `--resume`, the reproduction wrapper rejects a training output directory
that already contains `.pt` checkpoints. Resume uses `--stage train` so it does
not unnecessarily rerun cache export/finalization. `--resume` refers to a world-model
checkpoint, not the S-VAE checkpoint. Add `--dry-run` to any workflow command to
print all planned commands without starting Python or creating files. Use `--help`
for remaining options; the wrapper does not accept arbitrary native trainer flags.

To retain console output, run from the repository root and wrap the selected
command with standard shell logging, for example:

```bash
mkdir -p logs
set -o pipefail
PYTHONUNBUFFERED=1 bash scripts/reproduce_world_model.sh --stage train \
  --cache /path/to/finalized_cache --gpus 8 \
  2>&1 | tee -a logs/world_model_svae_d16_cam_high_100k.log
```

## Lower-level world-model training

```bash
bash scripts/train_world_model.sh \
  --cache /path/to/robotwin_svae_d16_cache \
  --gpus 8 \
  --output outputs/world_model_svae_d16
```

The default YAML is `configs/robotwin/dynamics/svae_d16.yaml`: cam_high/view 0,
108x16 tokens, four history frames, 16 prediction frames, batch size 32,
learning rate 6e-5, BF16 and a 100,000 optimizer-step budget. W&B and RGB decoder
training are disabled. The launcher uses the repository's Accelerate config,
not the user's global launch config, and enables multi-GPU mode when `--gpus > 1`.

Use `--config PATH` for another dynamics recipe and a separate `--output`.
Native overrides use the trainer's underscore spelling, e.g. `--batch_size 16`.
Change `training.max_train_steps` in a copied YAML for a smaller step budget;
`--num_epochs` alone does not shorten the run while this step budget is set.
Resume is explicit:

```bash
bash scripts/train_world_model.sh \
  --cache /path/to/robotwin_svae_d16_cache \
  --gpus 8 --output outputs/world_model_svae_d16 \
  --resume /path/to/world_model_checkpoint.pt
```

The trainer saves the resolved YAML and checkpoints under `--output`. For a
concurrent run choose another `--port` (default 29500) and output directory.

## Shared launcher controls

- `--help`: wrapper usage, without importing training dependencies.
- `--dry-run`: print the shell-escaped command; do not launch Python.
- `--python /path/to/python`: override the default `venv/bin/python`; `PYTHON_BIN` is also accepted.
- `PPU_SDK_SETUP`: SDK setup script, default `/usr/local/PPU_SDK/envsetup.sh`.
- `--gpus N`: local process count; `NUM_GPUS` is the fallback (default 8).
- `CUDA_VISIBLE_DEVICES`: choose visible devices before running either script.

The training launchers use `exec`, so training signals and exit status propagate directly.
They run in the foreground; redirect stdout/stderr or use the job manager of your
choice when starting a long run. Multi-node launching is outside these wrappers.

## 4. World-model validation and comparison

Use the same S-VAE checkpoint/cache, evaluation seed, batch size, episode selection,
and policy bundle for both checkpoints. The three-camera S-VAE supplies the codec;
world-model evaluation selects **cam_high (view 0)** only.

First evaluate the saved training configuration on its full validation split:

```bash
bash scripts/run_python.sh scripts/validate_world_model.py \
  --checkpoint /path/to/world_model/checkpoint_step_0100000.pt \
  --cache /path/to/robotwin_svae_cache \
  --batch-size 32 --seed 3047 --device cuda:0 \
  --output outputs/evaluation/validation.json
```

The script loads the adjacent `config.yaml`, uses fixed validation windows, resets
Python/NumPy/Torch RNG to `seed + batch_index`, and reports sample-weighted loss
plus batch-mean loss. Keep batch size fixed for a comparison: the shortcut/denoising
objective is stochastic. Compare the raw `sample_weighted_mean.flow` across checkpoints.
The total `loss` uses the checkpoint-specific EMA RMS normalization and is not on a
shared scale; normalization metadata is included in the report. This is an offline
objective check, not a task-success metric.
The current d16 recipe has 500 validation episodes and 2,058 windows (4+16 frames,
interval 3, window stride 60, tail alignment).

Then measure autoregressive feature/action fidelity on all 50 tasks, one success
and one failure episode per task. This bounded acceptance run predicts 16 frames
from one real frame, with 16 sampling steps per predicted frame:

```bash
bash scripts/evaluate_world_model.sh \
  --cache /path/to/robotwin_svae_cache \
  --world-model-checkpoint /path/to/world_model/checkpoint_step_0100000.pt \
  --codec-checkpoint outputs/svae_d16_3views/best.pt \
  --qwen-model /path/to/Qwen3-VL-4B-Instruct \
  --raw-data-root /path/to/robotwin-success-fail-lerobot \
  --policy-bundle /path/to/exported_qwenoft_bundle \
  --max-episodes 100 --max-steps 16 --device cuda:0 \
  --output outputs/evaluation/candidate.json
```

Omit `--policy-bundle` for feature-only evaluation. Check external paths in the
exported policy bundle configuration before moving it to another environment.
The launcher defaults to **full trajectories** (`--max-steps 0`); the explicit
`--max-steps 16` above is a short-horizon acceptance test, not full-trajectory or
closed-loop paper reproduction. To assess long horizons, rerun both checkpoints
with `--max-steps 0` and distinct output files. No intermediate ground-truth reset
is used. `--help` exposes the underlying evaluator options.

Run the same command for the historical checkpoint, changing only checkpoint,
output path, and optionally device. Then compare completed reports:

```bash
bash scripts/run_python.sh scripts/compare_world_model_evaluations.py \
  --baseline outputs/evaluation/history.json \
  --candidate outputs/evaluation/candidate.json \
  --output outputs/evaluation/comparison.json
```

The comparison rejects mismatched protocols, episode identities, prediction horizons,
and action-query counts. It writes JSON and Markdown with task macro means and paired
task-bootstrap intervals. These intervals measure task variation, not variation over
training seeds. Raw feature baselines, decoded-GT references, codec reconstruction,
and simulator task-success rates are different experiments and must not be mixed.

## Action-error diagnosis

`diagnose_world_model_actions.py` runs paired checkpoints on one device with
identical per-step noise, repeated policy inputs, codec-GT controls, autoregressive
rollout, teacher-forced rollout, and GT-proprio controls. Supply the same data,
codec, Qwen and policy arguments as evaluation, with `--baseline`, `--candidate`,
`--selection /path/to/evaluation.selection.json`, and `--output /path/to/diagnosis`.
Use `--seeds 3047,3048,3049` and select task names with `--tasks` (comma-separated).
These seeds control inference noise, not independent training runs. Diagnostic
teacher-forced and GT-proprio modes must not be reported as open-loop results.

`diagnose_original_rng.py` accepts native open_loop arguments and reproduces the
three diagnostic tasks at their original full-selection RNG offsets. Use the
original `--max-episodes 100 --max-steps 16 --views 0` settings. It verifies the
skipped-noise replay against actual generation at every evaluated step.

`audit_policy_feature_interface.py --diagnosis-run /path/to/diagnosis/invocation.json --output /path/to/policy_interface.json --max-frames 16` compares merger injection with
native QwenOFT with/without DeepStack, and can evaluate saved diagnostic predictions
against the full native-policy reference. The diagnostic script writes `invocation.json` with a `command` list
recording its invocation. No ground-truth DeepStack is injected into
the model-prediction branch. See [action metric semantics](../docs/evaluation.md#meaning-of-the-action-metric).

## Controlled original-versus-release training

`compare_training_implementations.py` runs each checkout's native training entrypoint
and trainer in separate processes. Use it to check a refactor under matched training
conditions; it does not compare already-trained checkpoints or measure task success.
The original checkout is required and is read without modifying its source files.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash scripts/run_python.sh scripts/compare_training_implementations.py \
  --original-root /path/to/TokenWorld \
  --config outputs/world_model_svae_d16_cam_high_100k/config.yaml \
  --output outputs/controlled_training_single \
  --steps 8 --processes 1 --seed 3047
```

For a two-device DDP comparison, use `CUDA_VISIBLE_DEVICES=0,1`, `--processes 2`
and a new output directory. Each implementation runs sequentially on the same devices.
The default eight steps with seed 3047 exercise both ordinary denoising and shortcut
training. The configured batch size (32 in the released recipe), bf16, native Adam,
EMA updates and eight DataLoader workers are retained. Only the short step budget,
fixed seeds, output locations, disabled W&B/resume and checkpoint frequency differ
from the source configuration. DataLoader and training-noise RNG are controlled
separately; noise differs per DDP rank but is matched across implementations.

The script checks native initialization, explicitly loads the same initial model
into the release run, then fingerprints every effective input batch, loss, gradient,
model parameter/buffer, and optimizer tensor. It also compares RNG consumption and
checks finiteness. Observers do not replace the native forward, backward, or optimizer
updates. Any byte-digest mismatch fails; no tolerance is used. `comparison.json`
lists all checks, and per-rank traces identify the first divergent stage. Initial
and final checkpoints are retained. The output directory must not already exist.
This bounded test does not establish equivalence of historical training trajectories,
eight-device/multi-node execution, or other hardware/software environments.

To test the native resume loaders too, add
`--resume-checkpoint /path/to/shared/checkpoint.pt` and choose another fresh output
directory. `--steps` then means **additional optimizer updates**. Both implementations
restore the same model, Adam moments and saved RNG state through their own loader;
`resumed.rank*.json` verifies those restored states before the first forward pass.
The fixed per-step training noise still applies. A matched resume test does not
reconstruct the historical DataLoader worker states missing from older checkpoints.

## RGB decoder entrypoint

Use `bash scripts/train_rgb_decoder.sh --cache PATH --data-root PATH --gpus 8`.
The default codec is `outputs/svae_d16_3views/best.pt`, output is
`outputs/rgb_decoder_svae_d16_cam_high`, and the repo venv/PPU SDK load automatically.
This freezes S-VAE, decodes d16 tokens to 2560-D merger features, and trains a
cam_high-only 224×224 RGB decoder. Global batch 32, 500 epochs and Adam/cosine
3e-4 → 1e-6 match the original recipe; eight devices get four trajectories each.
See the [full recipe, current-storage command, storage requirements and resume instructions](../README.md#rgb-decoder-training-robotwin-cam_high).

- `--check-only`: read-only input/config check and approximately 563-GiB cache estimate.
- `--smoke-test`: bounded one-epoch run, isolated under `OUTPUT_smoke`.
- `--dry-run`: show the distributed command without loading the environment.
- `--resume PATH`: restore an RGB decoder checkpoint; total epochs stay at 500.
- `--keep-feature-cache`: retain generated features after successful training.

The first process prepares LPIPS/AlexNet weights in the shared `TORCH_HOME` before
distributed launch. The generated feature cache is retained on failure and removed
after success by default. Policy/DeepStack behavior is unchanged.

## RGB replay verification

`replay_world_model.sh`, `evaluate_closed_loop.sh`, and `render_replay.sh` provide recorded-action replay, learned-world-model QwenOFT closed loop, and matched RGB comparison. See [protocol and commands](../docs/evaluation.md#rgb-replay-and-learned-world-model-closed-loop). These do not execute the RoboTwin simulator or automatically score task success.

Closed-loop diagnostics: `diagnose_closed_loop_policy.py` compares policy input interfaces; `diagnose_closed_loop_rollout.py` runs isolated binary-gripper or native-RGB interventions. See the evaluation protocol document. The production closed-loop wrapper now defaults to binary gripper inversion; `--gripper-mode legacy_minmax` reproduces the historical bug.


## Experimental DeepStack feature adapter

`train_deepstack_adapter.sh` runs the three-task, cam_high GT-latent pilot with the release venv. It freezes the existing S-VAE and policy, trains only a three-stream adapter, checks native RGB/injected-feature parity, and automatically evaluates held-out policy actions. World Model weights are not updated. This is an exploratory representation experiment, not a closed-loop benchmark or a default interface change.

```bash
cd /path/to/Token-World
bash scripts/train_deepstack_adapter.sh --output outputs/deepstack_adapter_pilot_20261031 --device cuda:0
```

Use a fresh output path for another run. See [experiment protocol](../docs/deepstack_adapter.md) for data splits, targets, metrics and limitations.


## Full RoboTwin DeepStack adapter pipeline (PPU)

Export all 50 tasks on eight PPUs, train the adapter on one PPU, evaluate it, then automatically remove this run's temporary feature files:

```bash
bash -lc 'cd /path/to/Token-World && bash scripts/reproduce_deepstack_adapter.sh --gpus 8 --output outputs/deepstack_adapter_all50_20261031'
```

Submit once on **one eight-PPU node** with shared storage. The release-relative `venv` and PPU SDK are loaded automatically. Defaults: all 4,499 training episodes (8 sampled frames each), 500 held-out validation episodes (4 frames each), cam_high, batch 32, 30 epochs / 33,720 updates. Temporary feature storage is approximately 79.1 GiB. Failures preserve features for recovery; successful training plus final evaluation triggers narrowly scoped cleanup. Checkpoints, metrics, configuration and manifests remain. Re-submit identical arguments to continue an interrupted run.

The small three-task adapter improved offline policy agreement but has **not** restored reliable learned-WM closed-loop behavior; full-data training must still be validated in closed loop. See [full protocol](../docs/deepstack_adapter_full.md) for recovery, cleanup, evaluation and current evidence.

## Public minimal inference

After following the root README installation and checkpoint download steps:

```bash
bash scripts/run_python.sh scripts/minimal_inference.py
```

The bundled example and all required model configuration are public. Full dataset, policy and simulator workflows still require the external assets listed in the root README.
