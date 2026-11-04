# Representation ablations

| Paper experiment | Required variants | Current preparation |
| --- | --- | --- |
| Table II codec reconstruction/modelability | 8, 16, 24, 32, 48, 96, 128 channels | Codec trainer accepts dimensions; SEC/LDS diagnostics retained |
| Table III dynamics dimensions | 8, 16, 32, 48 channels | Derived world-model configs supplied |
| Table IV representation | S-VAE, raw Qwen, SDXL-VAE | S-VAE/raw cache and training paths migrated; matched-backbone SDXL path unresolved |

Vary channel dimension while preserving the 9 x 12 grid. Do not substitute
the older spatially pooled PS-VAE configurations. Codec evaluation reports
reconstruction fidelity separately from world-model prediction fidelity.

```bash
python -m tokenworld.evaluation.representation \
  --cache /path/to/svae_cache --split val --view 0 \
  --output outputs/representation_d16.json
```

SEC is implemented as the fraction of orthonormal 2-D DCT energy above
`u+v > rho*(H+W-2)`; LDS is mean four-neighbor cosine minus mean cosine for
pairs at Manhattan distance at least `far_dist`. The defaults are rho=0.5
and far_dist=6, inherited from the available diagnostic implementation.
Confirm their exact use, dataset and aggregation in Table II before attaching
published values to this recipe.

Table III's d16 numbers differ from Tables I/IV. Its evaluation subset,
horizon and reference features must be recovered. Do not claim these derived
configs alone reproduce Table III. The available SDXL script uses IRASim
model imports and several missing helpers; it is archived outside the release as a migration candidate, not relabeled
as the paper's matched TokenWorld backbone.
