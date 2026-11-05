# Token-World

**Token-World: World Modeling in Vision-Language Model Token Space for Robot Manipulation**

[Paper](https://arxiv.org/abs/2610.00575) · [Project website](https://chuyaofu.github.io/Token-World/) · [Video](https://youtu.be/TCgozpMydts)

Token-World predicts action-conditioned dynamics in the visual token space of a frozen vision-language model. A separately trained semantic VAE (S-VAE) compresses each of 108 spatial Qwen tokens from 2,560 to 16 channels. A DeepStack adapter connects predicted tokens to the frozen QwenOFT policy. An auxiliary RGB decoder visualizes predictions.

This release provides the RoboTwin training and evaluation code, fixed data split, and reproduced checkpoint configuration. The supported training platform is **PPU with Python 3.12 and SDK PyTorch 2.6.0**. NVIDIA CUDA installation remains unverified.

## Model configuration

| Component | Released configuration |
|---|---|
| Visual encoder | Qwen3-VL-4B-Instruct, frozen |
| S-VAE | Three cameras, 108 tokens per camera, latent dimension 16 |
| World Model | cam_high, 100,000 updates, 50 Hz |
| Temporal context | KV within each segment, GRU hidden state across segments |
| Policy interface | Frozen QwenOFT, predicted merger plus three adapter-predicted DeepStack layers, native pre-norm readout |
| RGB decoder | Original epoch-500 cam_high decoder |

The RGB decoder can distort small objects even when reconstructing observed features. Decoder improvement scripts and diagnostic results are provided separately. The reproduced RGB metrics below use the original epoch-500 decoder.

## Install

Clone the repository on a node with a compatible PPU SDK, Python 3.12 and its matching system torch/torchvision packages.

```bash
git clone https://github.com/ChuyaoFu/Token-World.git
cd Token-World
bash scripts/setup_env.sh
bash scripts/run_python.sh -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

If the default PyPI endpoint is unreachable, select a reachable mirror, for example:

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ bash scripts/setup_env.sh
```

The installer creates `venv/` beside `scripts/`. Launchers locate this environment relative to the checkout and load `/usr/local/PPU_SDK/envsetup.sh`. Override `PPU_PYTHON` and `PPU_SDK_SETUP` for another SDK installation. Rebuild the venv if the checkout path changes. Training commands below use eight PPUs on one node and should be submitted with `bash` in scheduler jobs.

## Checkpoints and data

[The release checkpoint manifest](configs/checkpoints.json) lists file sizes and SHA-256 values for the S-VAE, World Model, DeepStack adapter and RGB decoder, plus the required World Model `config.yaml`. Weights are mirrored on [Hugging Face](https://huggingface.co/ChuyaoFu/Token-World) and [ModelScope](https://modelscope.cn/models/Zellovo/Token-World). Download into this checkout with either provider.

```bash
# Hugging Face
bash scripts/run_python.sh -m pip install 'huggingface-hub>=0.30,<1'
bash scripts/run_python.sh scripts/download_checkpoints.py --repo-id ChuyaoFu/Token-World

# ModelScope mirror
bash scripts/run_python.sh -m pip install 'modelscope-hub==0.4.3'
bash scripts/run_python.sh scripts/download_checkpoints.py \
  --provider modelscope --repo-id Zellovo/Token-World --revision master

# Check all local weights and the required configuration without downloading
bash scripts/run_python.sh scripts/download_checkpoints.py --verify-only
```

Both mirrors contain the same original checkpoint bytes. The downloader verifies the pinned size and SHA-256 values before reporting success.

## Minimal inference (one PPU)

After installation and checkpoint download, run:

```bash
bash scripts/run_python.sh scripts/minimal_inference.py
```

The demo uses one CPU compute thread to avoid a PPU SDK multithreaded CPU RoPE initialization crash observed during a fresh-install audit. Device inference still runs on one PPU.

This uses the small [bundled RoboTwin example](examples/robotwin/README.md), with no dataset download, policy bundle, Qwen download, simulator, or shared storage required. It starts from one cached `cam_high` latent and the initial robot state, replays 30 recorded actions in two 15-frame segments, carries the GRU state across the boundary, and resets cross-segment KV. All four released models are loaded: WM predicts latents, S-VAE reconstructs merger features, the adapter predicts three DeepStack layers, and the RGB decoder produces images.

Outputs are `outputs/minimal_inference/prediction.gif`, `comparison.png` (recorded RGB versus WM decoded RGB at frames 15 and 30), `predictions.npz`, and `report.json`. This is a runnable inference smoke test, **not policy action inference, a closed-loop success test, or a paper metric evaluation**. It does not run the VLM encoder from RGB; the initial latent is included. The sample's original codec/cache provenance is documented alongside it.

## Assets needed for full reproduction

**The full LeRobot data export, latent cache/train statistics, QwenOFT policy bundle, simulator Python/Mesa runtime and baseline checkpoints are not included in the current model repositories.** The full training/evaluation commands below require these additional assets; they cannot currently be reproduced from only the four published model weights. Generic upstream RoboTwin data does not establish the exact episode mapping and split of this export.

The RoboTwin export contains **4,999 trajectories across 50 tasks**, including 2,500 successes and 2,499 failures. The fixed split contains 4,499 training and 500 validation trajectories. Preserve the episode identifiers in `configs/robotwin/splits/seed3047.json`. Raw observations use three named camera streams and a 50 Hz time grid. See [the data contract](docs/data.md) for the expected LeRobot layout.

Set these paths to your downloaded assets before running training or evaluation.

```bash
export QWEN_MODEL=/path/to/Qwen3-VL-4B-Instruct
export RAW_DATA_ROOT=/path/to/robotwin-success-fail-lerobot
export LATENT_CACHE="$PWD/outputs/cache/robotwin_svae_d16_3views_50hz"
export POLICY_BUNDLE=/path/to/tokenworld_qwenoft_success_cam_high_teacher
```

The policy bundle contains the pinned StarVLA implementation, configuration, normalization statistics and RoboTwin QwenOFT 50k weights. [Integration documentation](docs/integrations.md) describes its layout. Qwen, RoboTwin simulator assets and metric backbone weights have their own upstream distribution terms.

## Reproduce training

### 1. Train the S-VAE

The reproduced checkpoint uses all three cameras. The script also supports `--views cam_high` for the single-camera experiment.

```bash
bash scripts/train_svae.sh \
  --model "$QWEN_MODEL" --data-root "$RAW_DATA_ROOT" \
  --views all --gpus 8 --output outputs/svae_d16_3views
```

### 2. Export latents and train the World Model

```bash
bash scripts/reproduce_world_model.sh \
  --model "$QWEN_MODEL" --data-root "$RAW_DATA_ROOT" \
  --codec-checkpoint outputs/svae_d16_3views/best.pt \
  --cache "$LATENT_CACHE" --gpus 8 \
  --output outputs/world_model_svae_d16_cam_high_100k
```

Export preserves all three cameras. The dynamics recipe selects cam_high and uses posterior means, training-split per-view normalization, 4 history frames, 16 prediction frames, learning rate 6e-5 and 100,000 optimizer updates. With eight PPUs, batch size 32 per process gives global batch 256. `--stage train` reuses a finalized cache. Use a fresh output directory or supply `--resume` for an existing run.

### 3. Train the DeepStack adapter

```bash
CACHE_ROOT="$LATENT_CACHE" \
bash scripts/reproduce_deepstack_adapter.sh \
  --gpus 8 --output outputs/deepstack_adapter_all50
```

Eight workers extract frozen visual features, then one PPU trains the adapter. The pipeline verifies completion before removing its own temporary feature shards. Use `--keep-features` to retain them. Set `ADAPTER_CHECKPOINT` to the resulting `adapter_best.pt` when evaluating a newly trained adapter.

### 4. Train the RGB decoder

```bash
bash scripts/train_rgb_decoder.sh \
  --cache "$LATENT_CACHE" --data-root "$RAW_DATA_ROOT" \
  --codec-checkpoint outputs/svae_d16_3views/best.pt \
  --gpus 8 --output outputs/rgb_decoder_svae_d16_cam_high
```

The default recipe runs 500 epochs. [Decoder diagnostics](scripts/decoder_improvement/README.md) compare raw merger inputs, S-VAE reconstructions and predicted World Model inputs using fixed images and object crops.

## Evaluate

### Open-loop feature and policy-action fidelity

```bash
bash scripts/evaluate_table1_robotwin.sh 8
```

The fixed balanced100 panel covers all 50 tasks with one success and one failure per task. It initializes from the real first frame, follows recorded dataset actions, and scores all 22,152 predicted frames. Each frame produces 50 normalized 14-dimensional policy actions. The default interface compares World Model predictions through the adapter against the same frozen policy directly observing the matching real RGB frame.

| Release configuration | Feature cosine ↑ | Feature NMSE ↓ | Action cosine ↑ | Action NMSE ↓ |
|---|---:|---:|---:|---:|
| Native adapter interface | 0.7731 | 0.4573 | 0.9598 | 0.0782 |
| Historical merger-only interface | 0.7731 | 0.4573 | 0.9406 | 0.1220 |

The two action protocols have different reference actions. Use `ACTION_INTERFACE=merger_only` for the historical paper interface. Cosine is averaged over 2,560-dimensional feature-token rows or 14-dimensional action rows. NMSE is total squared error divided by total squared reference energy. The table reports pooled micro scores. `comparison.json` also contains task macro scores.

### RGB fidelity

The current-checkpoint RGB comparison covers 500 recorded trajectories and 54,222 common predicted frames at 25 Hz. Both methods start from the real first frame and follow recorded actions. Token-World generates at 50 Hz and uses matching source-frame indices for scoring. Metrics use full 224 × 224 images.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD-16 ↓ | FVD-32 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Token-World | 21.4452 | 0.88468 | 0.13082 | 26.81562 | 89.03667 | 205.43271 |
| Ctrl-World | 18.0481 | 0.82709 | 0.16395 | 7.15455 | 18.86399 | 40.90872 |

**455 of these 500 trajectories appear in the current Ctrl-World training sampler.** All 500 belong to Token-World's validation split. The following comparison uses the shared validation intersection of 45 trajectories across 28 tasks, comprising 4,675 common frames.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD-16 ↓ | FVD-32 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Token-World | 21.3994 | 0.88257 | 0.12961 | 40.59502 | 166.35873 | 415.64984 |
| Ctrl-World | 17.6846 | 0.81934 | 0.16648 | 21.10899 | 96.47354 | 227.97365 |

The smaller intersection has an uneven task distribution. FID/FVD depend on sample count and extractor details. See [RGB evaluation](scripts/compare_rgb/README.md) for fixed preprocessing, LPIPS/AlexNet, FID/Inception and FVD/I3D definitions, generation commands and checkpoint identities. The current Ctrl-World checkpoint has not been verified as identical to the historical paper checkpoint.

### World Model closed-loop videos

The policy receives a real first frame, then acts on World Model predictions through the adapter. RoboTwin executes exactly the same actions from the same initial state to provide a visual reference and physical success check. The 50-task, 10-valid-seeds-per-task evaluation produced 500 aligned videos and 231 simulated successes. See [closed-loop video reproduction](scripts/closed_loop_video/README.md).

Simulator evaluation requires a separately provisioned SAPIEN/RoboTwin runtime and renderer. PPU inference and CPU Vulkan rendering run in separate processes. See [simulator setup](scripts/robotwin_sim/README.md) for environment paths, assets and initial-state handling.

## Repository layout

- `tokenworld/` contains the codec, dynamics, decoder, data pipeline and evaluation code.
- `configs/` contains training recipes, PPU dependencies and the fixed split.
- `scripts/` contains training, replay, evaluation and diagnostics entry points.
- `tests/` contains regression checks.
- `docs/reproducibility/` preserves dated implementation and verification records.

Historical records describe the experiments at the time they were written. The results and checkpoint configuration on this page describe the current reproduced release.

## Licenses

Code retains the original MIT and per-file third-party notices. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. Model and dataset cards document the respective artifacts and upstream dependencies.
