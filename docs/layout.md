# Repository layout and migration

All installed Python code now lives under `tokenworld`. Root directories are
`tokenworld`, `configs`, `docs`, `scripts`, `tests`, and `licenses`.

| Component | Location |
| --- | --- |
| S-VAE training/full codec and decoder-only loader | `tokenworld/codec/` |
| World model and trainer | `tokenworld/dynamics/` |
| RGB decoder, training and image metrics | `tokenworld/decoder/` |
| RoboTwin dataset, video I/O and cache export | `tokenworld/data/` |
| VLA interfaces | `tokenworld/policies/` |
| Rollout, open/closed-loop evaluation and diagnostics | `tokenworld/evaluation/` |
| Shared I/O, distributed helpers, checkpoint compatibility | `tokenworld/utils/` |

## Entrypoints

The preferred training launchers are `scripts/train_svae.sh` and
`scripts/train_world_model.sh`; see [script usage](../scripts/README.md).
They wrap the module entrypoints below without changing the training logic.


| Former entrypoint | Current entrypoint |
| --- | --- |
| `dreamer4.train_dreamer_vlm` | `tokenworld.dynamics.train` |
| `dreamer4.train_decoder_robotwin` | `tokenworld.decoder.train` |
| `eval.evaluate_tokenworld_open_loop_fidelity` | `tokenworld.evaluation.open_loop` |
| `eval.closed_loop_oft` | `tokenworld.evaluation.closed_loop` |
| `eval.prepare_closed_loop_manifest` | `tokenworld.evaluation.prepare_manifest` |
| `tokenworld.codec.train` | unchanged; cam_high only |
| `tokenworld.data.export_svae`, `tokenworld.data.export_raw` | unchanged |
| `tokenworld.evaluation.aggregate`, `tokenworld.evaluation.representation` | unchanged |

Run `python -m <module>` or install the package first. Old CLI module names
are retired; the compatibility reader handles known old checkpoint object
paths without adding old top-level packages. Model state_dict parameter keys
are preserved. CPU fixture checks do not replace validation with the actual
paper checkpoints, whose selection remains deferred.

## Configurations and data

Dynamics recipes are in `configs/robotwin/dynamics/`; `svae_d16.yaml` selects
the dynamics representation, not S-VAE training. The codec trainer currently
uses CLI arguments documented in `training.md`, so no unused codec YAML is
provided. RGB training uses `configs/robotwin/decoder/rgb.yaml`.
The fixed split is `configs/robotwin/splits/seed3047.json`, and PPU constraints
are `configs/environment/ppu.txt`. Large data, weights and outputs stay
outside the tracked source tree.

## Archived preparation material

Unfinished baseline/SDXL candidates, internal review notes and raw verification
logs are archived outside the release. Integration status remains documented
in `integrations.md`; validated baseline adapters can be added later.
Stable data audits, source provenance and check summaries live in
`docs/reproducibility/`. Pytest disk caching is disabled for this checkout.
