# Source notices

- The dynamics/decoder core under `tokenworld/` derives from Phil Wang's Dreamer4 implementation.
  Its original MIT license is retained as `LICENSE`. The release packaging,
  RoboTwin-only entrypoint and import boundaries have been modified.
- `tokenworld/codec/vqvae.py` retains its Sonnet Authors attribution and Apache-2.0
  header. The Apache-2.0 license text is included under `licenses/`.
- `tokenworld/codec/shared_vision_token_codec.py` is copied from the StarVLA
  source used by the research workspace. It matches the exported bundle
  documenting commit `6de4eeb06eca82a1d851e8a7ae6fb0c5817ec792`. The bundle's
  MIT license is retained as `licenses/StarVLA-MIT.txt`.
- Codec training/cache helpers and the OFT adapter were extracted from the
  research IRASim checkout, which includes an Apache-2.0 license. That text
  is retained as `licenses/IRASim-Apache-2.0.txt`; adapted files carry notices.
- `tokenworld/decoder/metrics/lpips` derives from Sou Uchida's
  [lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch). Its BSD-2-Clause
  license is retained as `licenses/LPIPS-PyTorch-BSD-2-Clause.txt`.
  The local implementation averages scores across a batch and adapts torchvision
  loading. PerceptualSimilarity linear weights and ImageNet backbone weights are
  downloaded separately by the metric loader.

`docs/reproducibility/source_manifest.json` maps copied/extracted files to their original
repository-relative paths and source hashes. A repository-root license is
not a blanket replacement for per-file notices, data terms or model terms.
The checkpoint model card identifies the released artifacts. External backbones,
policies, data and simulator assets retain their own distribution terms.
