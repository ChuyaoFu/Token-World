# TokenWorld closed loop versus identical-action simulation

Run on a single node after manually requesting eight PPU cards:

```bash
cd /path/to/Token-World
env -u ROBOTWIN_TASKS NUM_PPUS=8 PER_TASK=10 SEED_START=100000 MAX_STEPS=0 \
  bash scripts/closed_loop_video/run_8ppu.sh \
  outputs/robotwin_closed_loop_vs_sim_all50x10
```

The 50 tasks match the RGB all500 evaluation task list. They use **new simulator
initial states**, not the recorded validation videos. Each task tries consecutive
seeds starting at 100000 and retains the first ten stable resets, regardless of
policy success. Unstable resets are retained as receipts and skipped. Exhausting
100 attempts or encountering a simulator/code/resource error fails the worker;
ordinary failures are not silently dropped. This produces 500 comparison videos.

TokenWorld receives only the first real head-camera RGB and joint state. Its
first action chunk uses the native RGB policy; later chunks query the frozen
expert through the current native pre-norm DeepStack adapter using predicted WM
latents. The simulator executes exactly the same physical joint commands. It
returns consumed-step counts and success flags, with no visual or proprioceptive
feedback after initialization. The trial stops on simulator success or the task's
native evaluation limit. Simulation success is recorded for this WM-generated
action sequence; it is not a second independent RGB-policy evaluation.
WM generation always uses the full planned block, independently of simulator
termination; a terminal partial block contributes only its executed prefix.

The current release 100k WM, epoch500 original decoder, S-VAE best.pt and
adapter_best.pt are used. Generation uses 16 denoising steps, up to 15 transitions
per block, GRU-only state across blocks, and policy queries every 50 commands.
At 50 Hz, chunk row zero is context, rows 1–49 are targets, and the last row is
repeated for command 50. Both videos take source indices 0,2,4,... at 25 Hz.
There is no temporal rescaling or interpolation. An odd terminal command has
no corresponding 25 Hz video frame. The first frame in both tracks is the same
real initial image; subsequent WM frames use S-VAE → merger → RGB decoding.

The existing simulator asset preparation supports only three tasks. This launcher
adds the full pinned RoboTwin object archive to shared storage (approximately
3.74 GB transfer on the first invocation, cached thereafter), and prepares a
separate node-local all-task stage. Archive extraction validates ZIP member CRCs.
It keeps the bundled CPU Mesa renderer and linear joint control at five simulator
steps per 50 Hz command. Before loading models, it checks reset and success
predicate initialization for every requested task. The open-laptop, scale and
cabinet success fields normally initialized by expert play_once are initialized
directly using the same formulas, without executing expert actions.

Each valid `<output>/<task>/<seed>/` contains:

- `comparison.mp4`: left TokenWorld, right simulation, timestamps at 25 Hz.
- `tokenworld.mp4`, `simulation.mp4`, and a seven-time thumbnail strip.
- `actions.npy`: identical physical commands sent to both paths at 50 Hz.
- `wm_contexts.npz`: complete planned WM contexts, offsets and consumed prefixes.
- `latents.npz`: normalized WM state at every source time including frame zero.
- `tokenworld_rgb.npz`, `simulation_rgb.npz`: RGB and exact source-frame IDs.
- `reset.json`, `generation.json`, `result.json`, and `physics.log`.

The same command resumes completed episodes. Existing receipts must match the
experiment contract, which includes checkpoint hashes, code hashes, seed and
length settings. Completed generation can resume video rendering without running
the models or physics. `summary.json` is written only after checking ten valid
comparison videos for every task. Use a different output folder for changed
settings. Earlier recorded-action RGB trajectories cannot serve as these closed
loop trajectories because the action sequences and initial states differ.

For an authorized two-card short diagnostic only:

```bash
NUM_PPUS=2 PER_TASK=1 MAX_STEPS=52 ROBOTWIN_TASKS='put_object_cabinet turn_switch' \
  bash scripts/closed_loop_video/run_8ppu.sh outputs/closed_loop_vs_sim_smoke
```

This launcher never submits a scheduler job and never starts the full experiment
as part of preparing the command.

Preparation validation on ali-dev: two independent PPU workers completed 52
commands each (cabinet and switch), including policy queries at steps 0 and 50.
Both paired videos contained 27 matching source timestamps. Head-only rendering
matched the original full-observation renderer exactly at source frames 0,2,4,6
under the same moving action sequence for both tasks. All 50 tasks passed a stable
reset, head-render and native success-predicate check; adjust_bottle and
dump_bin_bigbin used seed 100001 after unstable seed 100000. Evidence is in
`results/robotwin_closed_loop_video_20261103`. The 500-video generation has not
been started.
