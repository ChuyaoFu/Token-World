# RoboTwin evaluation with TokenWorld observations

## Release adapter on the fixed reset panel

```bash
bash scripts/evaluate_robotwin.sh 8 25 100000
```

The default `gru` mode carries only GRU state between segments. The launcher checks each initial RGB, physical reset and first action prefix against the archived panel. Cabinet object IDs come from the reference reset, so reruns reproduce the same object even when asset enumeration order varies. The default shared reference is `outputs/robotwin_wm_pair_20261101T080111Z`. Select another verified reset-mode baseline with `ROBOTWIN_REFERENCE` and, if needed, `ROBOTWIN_PANEL`. To create a fresh reference, run the paired command below with `ROBOTWIN_CACHE_MODE=reset`. Each trial writes `gru_cache.json`, and the task summary records `cache_mode`, `gru_successes` and `reset_successes`.

## Fresh paired simulator episodes

Run from the release root on a single shared-storage PPU node. The arguments are card count, predetermined seeds per task, starting seed, and an optional new output directory.

```bash
bash scripts/evaluate_robotwin_paired.sh 8 25 100000
```

The release uses `gru` by default. Temporal attention builds and reuses KV within each WM segment. At the boundary, the next segment takes the predicted latent and proprioception, carries only the GRU hidden state from just before the overlapping frame, and rebuilds KV from that frame. The same rule applies when a policy query starts a new action chunk. Each branch and each episode starts with fresh state.

Set `ROBOTWIN_CACHE_MODE=reset` to reproduce the historical baseline that clears both states at each boundary. Use `ROBOTWIN_CACHE_MODE=full` for the ablation that carries both states. Per-seed `pair.json` and per-task `summary.json` record `cache_mode`. Older paired receipts without this field used `reset`.

The three tasks are `put_object_cabinet`, `shake_bottle_horizontally`, and `turn_switch`. Workers shard (task, seed) pairs. Each pair uses two fresh `demo_clean` scenes. The scene's initial head RGB and 14-dimensional joint state initialize TokenWorld. Both branches use the native RGB policy once for an identical initial action chunk. Subsequent queries use only WM-predicted latents, either through the adapter with the native pre-norm readout or the RGB decoder and native visual encoder. Each branch evolves its own WM state and uses matching noise seeds at equal simulation times. The physics process returns only executed action counts and success flags after initialization.

Each 14-D physical command goes both to the simulator and, as the destination action, to the WM. Chunk row 0 is context. The next 50 commands use rows 1–49 and repeat the last row at the boundary. Each command spans five 250 Hz simulation steps. World Model blocks have at most 15 transitions and use 16 denoising steps. Maximum evaluation lengths are the task's own limits, 700/700/400. This matches the 50 Hz release data protocol.

This is the PPU-compatible benchmark setup with SAPIEN CPU Mesa raster rendering, MPlib and the bundled linear joint controller. It uses predetermined consecutive seeds without expert feasibility filtering. Task success comes from RoboTwin's own `eval_success`. Cabinet checks placement and release, horizontal bottle shaking checks bottle height, and switch checks the joint angle. Treat the result as a paired evaluation under this protocol.

The launcher uses the shared release weights and policy bundle. Each node needs the simulator Python/Mesa runtime installed separately, with `ROBOTWIN_SIM_RUNTIME` explicitly set to its root. This runtime is not in the model downloads, Python 3 with requests, network access to the fixed RoboTwin asset revision on hf-mirror.com for first-time setup, and local scratch space. The default local stage is `/tmp/tokenworld-robotwin-paired-v1`; override with `TOKENWORLD_SIM_STAGE`. Preparation serializes node-local downloads. The object ZIP is a sparse range download with member CRC validation. The embodiment archive is SHA-256 verified. Model and simulator Python environments run in separate processes.

Run logs are `rank*.log`. For every completed pair, `<task>/<seed>/pair.json` records success, termination step, model paths, and query indices. The directory also contains both action arrays, the rendered initial images, and the physical reset record. A mismatch in physical reset, initial RGB, or shared first action prefix stops the comparison. An explicitly unstable reset, reproduced in both branches, is recorded as invalid. Per-task `summary.json` requires a receipt for every requested pair and reports the valid-pair denominator, invalid seeds, and discordant counts. Use a new output directory for a new invocation.

Before preparing assets, the launcher verifies and unpacks the shared CPU renderer archive at `outputs/robotwin_sim_runtime/cpu-renderer-v1.tar.gz` into node-local scratch. The simulation process uses the bundled Vulkan loader and Mesa Lavapipe ICD. A SAPIEN camera must produce a nonblank RGB image with those libraries before any policy workers start. Inspect `renderer.log` in the run directory for the result or the driver load error.

## Full temporal cache comparison

The full-cache variant runs the adapter with first-frame-only world-model feedback. It carries all temporal attention keys and values and the GRU hidden states between action blocks, including policy-query boundaries. Its time positions count from episode reset. Each block advances only the executed targets, and each episode gets a new cache. The final terminal action block is scored by physics without generating an unused next policy observation.

From the release root, run the three-task comparison on one eight-card node:

```bash
bash scripts/evaluate_robotwin_full_cache.sh 8 25 100000
```

For cabinet only, set `ROBOTWIN_TASKS=put_object_cabinet`. The launcher requires a completed reset-cache reference run containing the reset records, first action arrays and paired outcomes. Set `ROBOTWIN_REFERENCE` to that run and optionally `ROBOTWIN_PANEL` to its verified outcome panel. The shared experiment defaults to the 20261101 paired run. Each valid case must match its archived physical reset, image and initial action prefix. Archived unstable resets are omitted from both denominators. The per-seed receipt records both outcomes, the highest reached KV size, peak allocated and reserved device memory, and model generation time. `summary.json` includes paired gains and losses. To resume an interrupted individual worker, rerun its Python command with the same output and reference after the old worker has exited.
