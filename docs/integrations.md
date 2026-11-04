# Integration audit

Unfinished source candidates are archived outside the release repository.
Only adapters that pass protocol and runtime checks will be added under
`tokenworld/evaluation/baselines/`. No baseline reproduction is claimed yet.

| Required integration | Source found | Remaining work |
| --- | --- | --- |
| StarVLA OFT | Exported bundle and RoboTwin token-injection runner | Select artifact/config; validate RGB-vs-token policy equivalence |
| StarVLA PIv3 / GR00T | Framework implementations exist in the StarVLA bundle | Recover RoboTwin-specific state/noise/action adapters and configs; validate each head |
| Ctrl-World | Model adapter and RGB replay runner | Resolve legacy multi-view vs cam_high settings; match paper dataset and timing |
| WorldGym | Model adapter and standalone RoboTwin replay script | Replace inconsistent copied metadata and fixed paths; unify action-expert evaluation and timing |
| IRASim | External research checkout | Recover full RoboTwin baseline entrypoint and exact settings |
| SDXL-VAE ablation | Partial fixed50 script | Missing episode helpers; matched TokenWorld-backbone recipe not yet identified |
| RoboTwin reference environment | Exported bundle/source available externally | Pin environment/assets and initial scene/seed contract for the three policies |

The recovered OFT helper assumes the bundle layout:

```text
bundle/
  code/starvla/starVLA/...
  model/config.full.yaml
  model/checkpoints/*.pt
  model/dataset_statistics.json
```

The observed StarVLA provenance is commit
`6de4eeb06eca82a1d851e8a7ae6fb0c5817ec792` from the source bundle. The codec
copied into `tokenworld/codec` matches that bundle byte-for-byte. This does
not certify the same commit for all three final RoboTwin policy checkpoints.
External runtimes remain separate to avoid conflicts between their top-level
`models` packages and dependency versions.
