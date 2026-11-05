# External integrations and asset availability

The current public model repositories contain four Token-World checkpoints and the World Model configuration. The self-contained [minimal inference example](../examples/robotwin/README.md) requires only those files and this repository.

The following assets remain external and are **not distributed by the current checkpoint download script**:

| Workflow | Additional required assets |
| --- | --- |
| Train S-VAE / export latents | Qwen3-VL-4B-Instruct and the exact RoboTwin LeRobot export with episode IDs matching our split |
| Train / evaluate World Model | The exported S-VAE episode cache, cache specification, and training normalization statistics |
| Policy action metrics / closed loop | QwenOFT policy bundle with its code, 50k weights, configuration and action statistics; frozen Qwen weights |
| Simulator success evaluation | RoboTwin source/assets and a compatible separate SAPIEN/Mesa Python runtime (`ROBOTWIN_SIM_RUNTIME`) |
| RGB baseline comparison | Ctrl-World source, weights and separate Python environment (`CTRL_PYTHON`); metric backbones |

These workflows were validated in the research environment. Full public reproduction remains incomplete until the exact additional artifacts and acquisition instructions are released. Setting a path variable does not download or create the asset.

The QwenOFT helper expects this bundle layout:

```text
bundle/
  code/starvla/starVLA/...
  model/config.full.yaml
  model/checkpoints/*.pt
  model/dataset_statistics.json
```

The observed StarVLA source provenance is commit `6de4eeb06eca82a1d851e8a7ae6fb0c5817ec792`. Other policy heads and baseline integrations are research utilities, not additional ready-to-download public checkpoints. Keep external runtimes separate to avoid package and dependency conflicts.
