# RoboTwin: current Token-World / Ctrl-World RGB comparison

Run from `/path/to/Token-World` on PPU.
The user selected comparison at **shared native 25 Hz physical times**, without temporal interpolation. Token-World continues to generate at 50 Hz and only its matching even source frames are scored.

## Fixed inputs and outputs

- Full validation is **all500**: 50 tasks, five success and five failure episodes per task (250 success +250 failure). `prepare_panel.py --selection all500` selects every validation record from the frozen cache index and validates actions, videos and frame indices against both models.
- `scripts/table1_robotwin/panel.json` remains the separate **balanced100 Table I subset**, one success and one failure episode per task. Use `--selection balanced100` when reproducing that fixed subset.
- All500 contains **108,677 Token-World prediction frames at50Hz and54,222 common prediction frames at25Hz**, excluding conditioning frame0. The separate balanced100 subset contains22,152 and11,058 frames respectively. Odd final source frames are retained in Token-World artifacts but have no native Ctrl prediction and are not interpolated.
- Both initial observations refer to the same original cam_high frame 0. Model-specific frozen encoders retain their native preprocessing. Exact source-video SHA256 equality and destination-aligned recorded action/state equality are checked before generation.
- Token-World: release checkpoint `outputs/world_model_svae_d16_cam_high_100k/checkpoint_step_0100000.pt`, 16 denoising steps, 15 destination frames per segment, intra-segment KV and **cross-segment GRU only**. Only the final predicted latent and predicted proprio feed the next segment. S-VAE `outputs/svae_d16_3views/best.pt`; RGB decoder **original epoch500**, `outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt`.
- Ctrl-World: `/path/to/Ctrl-World/checkpoint-100000.pt`, single cam_high, 50 diffusion steps, native 25 Hz, 5 generated frames per call with 4-frame advance, native history indices `[0,0,-12,-9,-6,-3]`. Only frame 0 is real visual context. The original overlap convention keeps four frames from non-final calls and five from the last; repeated boundary reconstruction is indexed at the same physical time. All scored outputs are predictions.
- Ctrl's SVD `added_time_fps=7` is the checkpoint's trained conditioning value, distinct from source trajectory frequency. Do not change it to 25 or 50 without a separate protocol.
- Actual checkpoint hashes, generation settings, sampled source frame IDs, recursive history indices and saved RGB hashes are retained. The current Ctrl checkpoint has **not** been proven identical to the weights behind the historical paper row.

## Metrics

The reference is the actual cam_high RGB at each scored source frame. Full images are resized to 224×224 using torch bilinear interpolation with `align_corners=False, antialias=True`; no spatial crop or temporal alignment optimization. Pixels are clipped, rounded to uint8 and scored identically on [0,1]. PSNR has data range 1. SSIM uses the bundled 11-pixel Gaussian-window implementation. LPIPS uses the bundled **AlexNet ImageNet1K V1 backbone + LPIPS v0.1** weights on [-1,1]; local backbone and linear-weight SHA256 values are included in every metric report.

Primary aggregation: arithmetic mean of per-frame PSNR/SSIM/LPIPS, with equal weight per scored frame. Episode macro means and success/failure breakdowns are separate. PSNR derived from global MSE is separately named and never substituted for mean frame PSNR. All metrics use the same cached trajectories.

Per-episode artifacts contain `rgb.npz` (uint8 RGB and original source frame IDs), `latents.npz` (float32 unnormalized WM latents) and `metadata.json`. The 50 Hz Token-World latents can be decoded by another RGB decoder later without running the WM again. `score.py` saves aligned GT/Token/Ctrl sheets at first, middle and final predicted timestamps.

## Current run and archive audit

Results root: `outputs/robotwin_rgb_compare_20261102`.

- `panel.json`: balanced100 immutable input manifest used for the pilot. The full500 manifest is `outputs/robotwin_rgb_all500_current_25hz/panel.json`.
- `smoke`: short initial 32-source-frame alignment check. Its imported historical Ctrl RGB is diagnostic only.
- `ctrl_checkpoint_probe.json`: numerical replay check against old archived Ctrl trajectories. Archive metadata names the current checkpoint path but has no generation-time checkpoint hash. The probe showed small nonzero pixel/latent differences, so **main comparison uses newly generated current-checkpoint Ctrl RGB**, not those archived predictions.
- `pilot/tokenworld`: full 50 Hz generation of eight fixed-panel episodes, four tasks × success/failure: adjust_bottle, open_laptop, put_object_cabinet, stack_blocks_two. Source prediction count: 1,926. This is a subset diagnostic, not the full 50-task result.
- `pilot_current/tokenworld`: symlink to the same Token-World artifacts, avoiding duplicate generation.
- `pilot_current/ctrlworld_25hz`: newly generated Ctrl trajectories with current checkpoint SHA256.
- `pilot_current/pilot_scores`, `pilot_current/pilot_report`: paired subset metrics and CSV after both models complete.

Archived replay reuse remains available in `import_ctrl.py` for explicit historical diagnostics; `run_8ppu.sh` always runs the current Ctrl checkpoint. Do not mix archive and current results in one table.

## Reproduce the eight-episode diagnostic

```bash
r=outputs/robotwin_rgb_compare_20261102
ids=0,16,39,45,50,66,89,95
bash scripts/run_python.sh scripts/compare_rgb/prepare_panel.py --output "$r/panel.json"
bash scripts/run_python.sh -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/compare_rgb/replay_tokenworld.py --panel "$r/panel.json" --output "$r/pilot" --episode-indices "$ids"
mkdir -p "$r/pilot_current"
ln -s ../pilot/tokenworld "$r/pilot_current/tokenworld"
PYTHON_BIN=/path/to/ctrlworld-venv/bin/python \
  bash scripts/run_python.sh -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/compare_rgb/replay_ctrlworld.py --panel "$r/panel.json" --output "$r/pilot_current" --episode-indices "$ids" --grid 25
bash scripts/run_python.sh -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/compare_rgb/score.py --panel "$r/panel.json" --artifacts "$r/pilot_current" \
  --output "$r/pilot_current/pilot_scores" --episode-indices "$ids" --grid 25
bash scripts/run_python.sh scripts/compare_rgb/summarize.py --panel "$r/panel.json" \
  --scores "$r/pilot_current/pilot_scores" --output "$r/pilot_current/pilot_report" --episode-indices "$ids" --grid 25
```

Generation skips completed episodes only when checkpoint hashes and protocol match exactly. Choose a fresh root for an independent reproduction; completed pilot episodes can be reused for all500 because shard count and panel membership do not affect episode RNG. `reuse_artifacts.py` verifies every episode input, records the original generation metadata and links the unchanged RGB/latent files into the larger panel. Only selection metadata is rebound.

## Full all500 job: manually submit on eight PPUs

The launcher never submits a scheduler job itself. It generates Token-World, generates Ctrl-World, scores both saved trajectories and validates exact **54,222-frame paired coverage**. The default selection is `all500`; `SELECTION=balanced100` explicitly selects the Table I subset. Ctrl uses the established legacy PPU venv with diffusers; Token-World and metrics use the release venv. Device visibility from the scheduler is preserved.

```bash
cd /path/to/Token-World
# Reuse the already completed8 pilot episodes; generate the remaining492.
DRY_RUN=1 bash scripts/compare_rgb/run_8ppu.sh outputs/robotwin_rgb_all500_current_25hz
bash scripts/compare_rgb/run_8ppu.sh outputs/robotwin_rgb_all500_current_25hz
```

Full output: `outputs/robotwin_rgb_all500_current_25hz/report/report.json`, `frames.csv`, `summary.md`. The pilot report remains under `outputs/robotwin_rgb_compare_20261102/pilot_current/pilot_report`. The full500 manifest and reuse of all16 model/episode artifacts (8 episodes ×2 models) have been checked; full generation has not been submitted.

For a fresh run, pass a new output root. `NUM_PPUS` controls 1–8 workers; `CTRL_PYTHON` overrides the Ctrl runtime. `REUSE_FROM` overrides the pilot artifact root; set it to an empty string to disable reuse. Avoid concurrently launching unrelated distributed collectives on the same PPU allocation.

## Completed pilot result

Eight freshly generated current-checkpoint trajectories, four tasks × success/failure,960 paired source frames at25Hz; no archived Ctrl RGB in this result. Values below are equal-weight means of per-frame metrics.

|Model|PSNR ↑|SSIM ↑|Alex-LPIPS v0.1 ↓|
|---|---:|---:|---:|
|Token-World release100k + original epoch500 RGB decoder|20.9222|0.87737|0.13887|
|Ctrl-World current single-view100k|17.2236|0.81455|0.17706|

These are four-task diagnostic results, not all500 or the balanced100 Table I result. All500 generation remains pending manual submission of the eight-PPU command above; it reuses the8 pilot trajectories and generates the remaining492. Analytical range/metric invariants passed (`metric_invariants.json`); exact paired-frame coverage, checkpoint/decoder hashes and absence of historical prediction reuse were checked in the pilot report.

## FID and FVD extensions

The same cached RGB now supports **FID2048, FVD16 and FVD32**. They are pooled distribution distances, not averages of per-frame or per-episode distances. Paired PSNR/SSIM/LPIPS remain separate measures of correspondence to the actual trajectory.

- FID: isolated `pytorch-fid==0.3.0`, TensorFlow-compatible FID Inception pool3 (2048D). Saved/GT RGB224x224 is scaled to[0,1], internally resized to299x299 by the reference implementation and normalized to[-1,1]. This is not generic torchvision ImageNet Inception or Clean-FID.
- FVD: the StyleGAN-V TorchScript conversion of TensorFlow I3D trained on Kinetics400;400-dimensional pre-softmax features. Input is contiguous float32 BCTHW in[0,255], detector flags `rescale=True, resize=True, return_features=True`; detector spatial size224x224.
- Videos use the existing native25Hz frame grid, starting at source frame2. Extract nonoverlapping complete16-frame and32-frame clips independently inside each trajectory. No real conditioning frame0, cross-trajectory concatenation, interpolation, repeated-frame padding or incomplete final clips. All500 trajectories have at least37 predicted native25Hz frames, so every trajectory contributes to both FVD variants.
- All500 sample plan:54,222 images for FID;3,134 clips for FVD16;1,449 clips for FVD32. Clip counts are not counts of independent trajectories: multiple clips within one episode are correlated.
- Covariance conventions: FID divides byN-1, matching `pytorch-fid`; FVD divides byN, matching StyleGAN-V feature statistics. `frechet.py` uses the equivalent low-rank nuclear-norm formula for undersampled pilots and a symmetric PSD eigendecomposition otherwise. Agreement with SciPy sqrtm on nonsingular test data, self-distance, translation and low-rank invariants passed.

Detector artifacts are cached under `outputs/pretrained/rgb_metrics`; package code is isolated under `outputs/rgb_distribution_metrics/python`. SHA256 values are pinned and checked against the authors' downloads:

- FID weights: `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.
- I3D TorchScript: `bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`.

Authors' downloads were obtained and hashed locally. The ali-dev cache was transferred through byte-identical public mirrors because the direct network route was slow; `mirror_downloads.json` records retrieval URLs and matching hashes. The canonical definitions are [pytorch-fid](https://github.com/mseitzer/pytorch-fid), [original Google FVD](https://github.com/google-research/google-research/tree/master/frechet_video_distance), and [StyleGAN-V FVD](https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py). Absolute scores from different feature extractors, interpolation schemes, clip lengths or sample counts should not be mixed.

The existing `run_8ppu.sh` command includes these metrics by default (`DISTRIBUTION_METRICS=1`). It preflights detector weights before generation. After paired scoring it extracts/caches features and writes `distribution_report/report.json`; those results are also linked into the main `report/report.json`. Use `DISTRIBUTION_METRICS=0` to run only paired pixel metrics.

For trajectories already generated, run only post-processing:

```bash
cd /path/to/Token-World
NUM_PPUS=8 bash scripts/compare_rgb/score_distribution.sh outputs/robotwin_rgb_all500_current_25hz
```

This launches no world model or RGB decoder. Eight pilot episodes already have cached real/Token/Ctrl features under `outputs/robotwin_rgb_all500_current_25hz/distribution_features`; the full job reuses them after RGB identity checks. Both feature-network extraction and numerical aggregation have run successfully on ali-dev PPUs. The first I3D integration required contiguous BCTHW input for the reference TorchScript `.view`; this was corrected without changing the detector or its preprocessing.

### Eight-episode pipeline diagnostic only

|Model|FID2048 ↓|FVD16 ↓|FVD32 ↓|
|---|---:|---:|---:|
|Token-World|76.30218|247.95627|671.33227|
|Ctrl-World|48.49981|213.66421|520.38569|

The pilot has only960 images,56 clips16 and27 clips32. Covariance estimates are rank-deficient and sample-biased; **these are pipeline diagnostic numbers, not an all500 result or a reliable overall model ranking**. Formal metrics remain pending the complete500-episode run. Full audit: `outputs/robotwin_rgb_all500_current_25hz/distribution_diagnostic/report.json`, including sample identities, feature/weight hashes and source generation contracts.
