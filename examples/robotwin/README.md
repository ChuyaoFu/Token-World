# Minimal RoboTwin inference sample

This is the first 31 frames (50 Hz) of the validation trajectory `robotwin_success/episode_001959`, task `put_object_cabinet`, camera `cam_high`. The small sample is distributed with this release for inference testing. RoboTwin is the upstream simulator; the recorded rollout is from our RoboTwin data export.

`sample.npz` contains one unnormalized initial S-VAE latent `[1,1,108,16]`, the initial 14-dimensional robot state, 31 absolute-qpos action rows (boundary plus 30 destinations), and training-split per-view latent/action normalization statistics. Continuous joints occupy indices `[0..5,7..12]`; the two grippers occupy `[6,13]`. No subsequent ground-truth latent is fed to the model. Three reference PNGs are for display only, resized to 224×224.

`sample.json` records hashes and source provenance. The cache used to train the released World Model was exported with the historical S-VAE checkpoint (SHA `95f8a7f…`), whereas the separately reproduced S-VAE released here has SHA `2eb38012…`. This demo retains the WM's original input/statistics contract and uses the released S-VAE to decode its outputs, matching the currently evaluated release pipeline. It is a loading/rollout smoke test, not evidence of an identical codec retraining or a quality benchmark. The full source cache is not required at runtime.
