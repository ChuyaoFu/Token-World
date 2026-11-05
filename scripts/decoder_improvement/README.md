# RoboTwin RGB decoder diagnosis and improvement

Remote checkout: `/path/to/Token-World`.
All commands below run from that checkout using the PPU environment in `scripts/run_python.sh`.
Results: `outputs/decoder_improvement_20261102/`. Baseline weights are never overwritten.

## Evidence and scope

- Baseline: `outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt`, SHA256 `9e85e03e3b855703b0f0e898cdbc7b3b314cece97a1f85e9674207c75813e104`.
- Pilot: four held-out episodes, two tasks × success/failure, four trajectory-wide frames each. WM features generated once with release100k, intra-segment KV and cross-segment GRU only; recorded actions and recursive predicted proprio/latent feedback. Reuse these exact features for all decoders.
- Diagnostic seed100014: four simulator images at steps0/50/150/250, scored separately. Its object is `048_stapler` model4. Training supplementation uses twelve other seeds, including stapler model0 and model5, and excludes model4. Seed100024/model0 soap supplies another held-out simulator case.
- Pilot training: 144 frames from24 train episodes. Supplementation adds96 simulator frames; no validation RGB is used in a gradient step. All training variants initialize from the original epoch500 decoder.
- Broad validation:100 diagnostic validation episodes (50 tasks × success/failure), four frames each; this diagnostic selection is separate from the formal Table I balanced100 panel and does not replace its scores.
- Formal Token-World/Ctrl-World evaluation is deferred while decoder work is prioritized.

## Findings established by code and measurements

1. Real merger→RGB already deforms the red stapler. S-VAE adds some error; WM adds trajectory/presence errors. Better RGB decoding alone cannot reconstruct an object absent from a predicted latent.
2. The CNN maps108×2560 tokens to a9×12 feature grid. It has residual convolutional blocks, a middle self-attention block and four nearest-neighbor+convolution upsampling stages, yielding144×192 before resizing to224×224. Per-token learned LayerNorm precedes the convolution. The output is tanh in[-1,1]. Original training uses MSE only.
3. Decoder training decodes **unnormalized cached16-D S-VAE tokens** into bf16 merger features. WM normalization must be inverted exactly once before S-VAE decoding; no latent-stat standardization is applied to the2560-D merger. A transpose of the spatial grid is a destructive negative control, not a repair.
4. Release and historical codec files have different serialization hashes but all82 state tensors are exactly equal (`input_audit.json`).
5. Policy and pretrained Qwen features on identical pixels match exactly in the measured four-image audit; observed image grid is1×18×24, spatial merge2 gives9×12. Old cache export uses PyAV/PIL bilinear resizing; Table I uses decord/tensor bilinear+antialias then uint8 PIL. Re-encoding the historical PIL path nearly reproduces cached bf16 latents (cosine0.9999967). Do not attribute this preprocessing difference to a different codec.

## Variants

- `candidate_v1`:120 steps, original144-frame train probe, 25% raw merger/75% cache-decoded merger, LR1e-5. Loss = MSE +0.1L1 +0.05Alex-LPIPS +0.1edge L1 +0.1GT-edge-weighted L1.
- `candidate_v2`:800 steps on the240-frame supplemented probe, 50% raw/50% S-VAE-reconstructed inputs, LR2e-5; same loss. Improves perceptual metrics and simulator appearance but loses some original-domain PSNR and does not improve stapler silhouette IoU.
- `candidate_v2_mse`: same data, steps, LR and input mix, MSE only. Separates data supplementation from the perception/edge loss. It gives much poorer LPIPS than v2 despite similar simulator PSNR.
- `candidate_v3_color`: v2 +0.02 generic dominant-color soft-Dice/foreground L1. No additional benefit; not selected.
- Architecture ablation: `models.py` retains the CNN and adds learned2D spatial position, zero-initialized merger injections at each scale, and a zero-initialized token-to16×16-pixel residual detail branch. Initial outputs must match the original CNN exactly. Architecture comparisons use an explicit sample RNG so both models see identical step-by-step training samples.

These are diagnostic training runs, not a production replacement or proof of complete shape recovery. Inspect crop evidence and broad validation before selecting weights.

## Metric contract

Full images:224×224, identical bilinear target resize, no crop; floatRGB in[-1,1]. PSNR uses data range1 after conversion to[0,1], computed per frame then arithmetic mean. LPIPS uses the bundled AlexNet backbone, torchvision ImageNet1K V1 weights and LPIPSv0.1 linear weights; exact local weight hashes are recorded in each report. Evaluate one image at a time and average equally across frames, with dataset validation and simulator seeds separated.

`object_rois.json` freezes identical crop locations for every model. Cabinet/alarm ROIs were chosen from GT images. Seed100014 crops use the largest GT red connected component plus8 pixels of margin; candidate outputs never determine the ROI. Crop LPIPS resizes the same crop to128×128 for adequate network support. Red-mask IoU is a color-threshold proxy, not an object segmentation ground truth; inspect it with the crop images and color error. Old coarse cabinet ROIs in exploratory reports must not be mixed with the final object ROIs.

Reports retain frame identities, source frame indices, all four decode-path metrics, checkpoint/probe hashes, fixed crop coordinates and per-frame predictions. The historical WM RGB shown for seed100014 is context from an earlier replay, **not** a newly decoded candidate WM output. Candidate WM decoder comparison is measured on the sixteen pilot WM feature probes.

## Reproduction

```bash
r=outputs/decoder_improvement_20261102
bash scripts/run_python.sh scripts/decoder_improvement/diagnose.py \
  --output "$r/pilot" --train-episodes 0 --val-episodes 4 --frames 4
bash scripts/run_python.sh scripts/decoder_improvement/diagnose.py \
  --output "$r/train" --train-episodes 24 --val-episodes 0 --frames 6 --skip-wm --device cuda:1
bash scripts/run_python.sh scripts/decoder_improvement/prepare_sim.py --run "$r"
bash scripts/run_python.sh scripts/decoder_improvement/make_rois.py \
  --probes "$r/probes_eval_extended.pt" --output "$r/object_rois.json"
bash scripts/run_python.sh scripts/decoder_improvement/train.py \
  --baseline outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt \
  --train-probes "$r/probes_sim_train.pt" --validation-probes "$r/validation_extended_guard.pt" \
  --output "$r/reproduce_v2" --steps 800 --batch-size 4 --lr 2e-5 --raw-fraction 0.5
bash scripts/run_python.sh scripts/decoder_improvement/evaluate.py \
  --probes "$r/probes_eval_extended.pt" --roi-json "$r/object_rois.json" \
  --decoder outputs/rgb_decoder_svae_d16_cam_high/decoder_epoch_0500.pt --output "$r/baseline_final"
bash scripts/run_python.sh scripts/decoder_improvement/evaluate.py \
  --probes "$r/probes_eval_extended.pt" --roi-json "$r/object_rois.json" \
  --decoder "$r/reproduce_v2/decoder_candidate.pt" --output "$r/reproduce_v2/evaluation"
bash scripts/run_python.sh scripts/decoder_improvement/compare.py \
  --baseline "$r/baseline_final" --candidate "$r/reproduce_v2/evaluation" --output "$r/reproduce_v2/comparison"
```

Use fresh output directories for preparation/training. Each diagnostic command runs in the foreground, so its exit status can be checked. Run Qwen loading and distributed-training checks separately; one concurrent DDP check received a native SIGSEGV, while isolated checks passed. The native crash cause has not been established.

## Eight-PPU job for manual submission

This launcher prints its exact command and uses all training episodes with on-the-fly frozen S-VAE decoding, rather than materializing a huge full merger cache.25% of updates draw from other simulator seeds; the remaining updates draw from the full original train split. No validation episode is sampled. Default:5000 steps, global batch32 RGB frames (8 ranks ×2 episodes ×2 frames), LR1e-5. It does not submit a scheduler job by itself.

```bash
cd /path/to/Token-World
DRY_RUN=1 bash scripts/decoder_improvement/train_8ppu.sh outputs/decoder_shape_full_8ppu_v1
# Submit the following inside the established eight-PPU scheduler job:
bash scripts/decoder_improvement/train_8ppu.sh outputs/decoder_shape_full_8ppu_v1
# After training, evaluate without regenerating any trajectories:
bash scripts/decoder_improvement/evaluate_candidate.sh \
  outputs/decoder_shape_full_8ppu_v1/decoder_candidate.pt \
  outputs/decoder_shape_full_8ppu_v1/broad_evaluation
```

`ARCHITECTURE=multiscale` selects the experimental architecture. Its checkpoints use the loader in `decoder_improvement.models`; the release production decoder loader has not been changed or deployed. The default remains the existing CNN until the architecture ablation has evidence of benefit.

## Completed broad validation (400 frames)

| Decoder | Cached S-VAE PSNR ↑ | Cached S-VAE Alex-LPIPS ↓ | Raw-merger PSNR ↑ | Raw-merger Alex-LPIPS ↓ |
|---|---:|---:|---:|---:|
| Original epoch500 |29.0199|0.05732|29.1315|0.05493|
| v1,120 steps |29.0506|0.05018|29.2058|0.04758|
| v2, supplemented,800 steps |28.7622|0.04672|29.0532|0.04402|

v1 is the conservative image-quality candidate: better average PSNR and LPIPS on this validation set, but no demonstrated repair of the red-stapler silhouette. v2 trades about0.26dB of cached-domain PSNR for lower LPIPS and much better simulator appearance. Keep these tradeoffs explicit; do not call either a completed shape fix.

## Architecture ablation, same training batches

The CNN control and multiscale variant both trained800 steps on the same240 probes, explicit sample RNG seed42, LR2e-5, batch4 and identical losses. Zero-initialized RGB parity passed with maximum error0 on raw and cache-decoded inputs. Parameters:98,685,955 →109,353,347.

| Architecture |400-frame S-VAE PSNR ↑|400-frame S-VAE LPIPS ↓|16-frame WM LPIPS ↓|Stapler raw-merger crop IoU ↑|
|---|---:|---:|---:|---:|
| Matched CNN control |28.7925|0.046744|0.09559|0.57763|
| Multiscale conditioning + detail branch |28.7953|0.046096|0.09447|0.56894|

The architecture produces a small perceptual improvement (~1.4% LPIPS), essentially unchanged PSNR, and no staple silhouette improvement. It remains an experiment. This does not establish that an architecture change is the main remedy, and does not rule out a better architecture after substantial training. The small overfitting probe cannot establish an information-theoretic limit of merger features.
