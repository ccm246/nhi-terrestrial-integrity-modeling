# Forest v2: canopy + human-pressure reference (second-machine instructions)

Reference sites = forest zone (Hansen canopy > 20 % on >= half of a 500 m cell, loss through 2018
removed, ESA WorldCover natural land) AND human modification <= 0.1 AND human influence < 0.08
(same tests as the grassland reference; scripts/tools/make_forest_reference_v2.py).
Responses: d02-d07 and d11-d19 (short vegetation height d08, tree cover d09 and non-tree vegetation
cover d10 are dropped). Predictors: the 20 abiotic bands d20-d39. Two variants per landscape:
`*_forestv2_predictors_only` (integrity map) and `*_forestv2_all` (vegetation as predictors, fit only).

Landscapes for tonight (reference km2 / forest km2): nepal_tal_2026 2,590 / 8,942;
adirondacks 12,274 / 22,076; seka 8,350 / 13,462; sabah 24,535 / 53,876.

## Steps (in the repo folder on the second machine)

```bash
git pull fork central-india-variants
source .venv/bin/activate
ulimit -n 4096
./run_forest_v2_pipeline.sh nepal_tal_2026
./run_forest_v2_pipeline.sh adirondacks
./run_forest_v2_pipeline.sh seka
./run_forest_v2_pipeline.sh sabah
```

Each run writes logs/<name>_forestv2/steps.log and, when finished, a DONE file in the same folder.
Run them in separate terminal windows to go faster (each has its own tile cache).

## Results to send back

```bash
zip -r forestv2_results.zip outputs/integrity_parameters outputs/reference_condition_inference outputs/samples -i "*forestv2*"
```
