# External integrations and asset availability

The public model repositories contain four Token-World checkpoints, the World Model configuration, and an optional QwenOFT policy bundle. The self-contained [minimal inference example](../examples/robotwin/README.md) requires only those files and this repository.

Asset acquisition for each workflow:

| Workflow | Additional required assets |
| --- | --- |
| Train S-VAE / export latents | Qwen3-VL-4B-Instruct from upstream; the existing [RoboTwin LeRobot dataset](https://modelscope.cn/datasets/Zellovo/robotwin-success-fail-lerobot) with authenticated access |
| Train / evaluate World Model | Generate the S-VAE cache and training statistics locally using `scripts/reproduce_world_model.sh --stage export` |
| Policy action metrics / closed loop | `scripts/download_policy.py` downloads the 50k policy, code/config/statistics; Qwen weights come from upstream |
| Simulator success evaluation | RoboTwin source/assets and a compatible separate SAPIEN/Mesa Python runtime (`ROBOTWIN_SIM_RUNTIME`) |
| Optional RGB baseline comparison | Obtain baseline source/weights separately; baseline checkpoints are outside this release scope |

Caches are derived artifacts, and baselines are optional comparison dependencies. Neither is a missing model upload. The simulator Python/Mesa environment remains a separate installation task; the existing simulator scripts require an explicitly configured runtime.

The QwenOFT helper expects this bundle layout:

```text
bundle/
  code/starvla/starVLA/...
  model/config.full.yaml
  model/checkpoints/*.pt
  model/dataset_statistics.json
```

The observed StarVLA source provenance is commit `6de4eeb06eca82a1d851e8a7ae6fb0c5817ec792`. The public archive includes the validated compatibility patches and preserves the source licenses. Other policy heads and baseline integrations are research utilities outside this checkpoint release. Keep external runtimes separate to avoid package and dependency conflicts.
