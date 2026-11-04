# Reproducibility evidence

This directory contains stable summaries for the current release candidate.
Raw development logs, old preparation reports and unfinished integration
sources are archived outside the release repository.

- [Experiment status](status.json): supported scope and remaining work.
- [Validation](validation.json): 22 CPU tests, open-loop regressions and 11
  CLI entrypoints passed. Fixed-seed CPU fixtures match the pre-migration
  parameter keys, tensors and outputs exactly for dynamics, S-VAE and RGB
  decoding. Four cases exercise legacy RGB checkpoint configuration/weight
  formats without importing the old packages.
- [Package audit](package_audit.json): the built wheel contains only the
  tokenworld package plus distribution metadata/licenses; imports pass
  outside the source checkout.
- [Data audit](data_audit.json): split and referenced-file checks for 4,999
  episodes; video/parquet content validation is still pending.
- [Source manifest](source_manifest.json): source paths and hashes for
  retained files, alongside the new release paths.

These checks do not constitute a GPU training run, validation of the actual
paper checkpoints or reproduction of published experimental numbers.
