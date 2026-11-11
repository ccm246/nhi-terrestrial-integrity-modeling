# Adirondack transfer: train on Adirondack reference sites, predict over a 264,000 km2 extent

Extent: config/aoi/adirondacks_large.geojson (79.0 to 71.5 W, 42.8 to 46.8 N; northwest Ontario and Quebec,
the St. Lawrence, Champlain and Mohawk lowlands). Forest zone (habitat codes 1 and 2): 178,151 km2,
where the models are applied. Two reference types, both inside the Adirondack Park only:

| Run | Reference sites | km2 |
|---|---|---|
| `adirondacks_large_hm` | canopy > 20 %, human modification <= 0.1, human influence < 0.08 (forest v2 reference) | 12,274 |
| `adirondacks_large_wild` | forest inside Park Agency Wilderness, Primitive and Canoe Area classes (no human-pressure test) | 5,072 |

Both use 50 km validation blocks and 5 folds (17 blocks hold hm reference, 12 hold wild reference).
Responses are the forest v2 list (d02-d07, d11-d19). 91 tiles per run.

## Steps (in the repo folder on the second machine)

```bash
git stash            # only if git complains about local changes
git pull origin central-india-variants
source .venv/bin/activate
ulimit -n 4096
./run_transfer_pipeline.sh adirondacks_large hm          # downloads the tiles, then fits and applies
./run_transfer_pipeline.sh adirondacks_large wild hm     # reuses the hm tiles (band d01 swapped), then fits and applies
```

Run them one after the other: the wild run needs the finished hm tile cache. Each writes
logs/adirondacks_large_<ref>/steps.log and a DONE file when finished.

## Results to send back

```bash
zip -r transfer_results.zip outputs/integrity_parameters outputs/reference_condition_inference outputs/samples -i "*adirondacks_large*"
```
Also send data/gee_cache_adirondacks_large_hm (the tiles) if you want similarity maps from this Mac.

## Rebuilding the inputs from scratch (this Mac only)

```bash
python -m scripts.tools.make_landscape_config adirondacks_large "Adirondacks large (transfer)" \
    config/aoi_sources/adirondacks_large_extent.geojson --ee-project charged-sled-403915 --buffer-km 0 --aoi-mode dissolve
python -m scripts.tools.classify_habitat adirondacks_large
python -m scripts.tools.make_transfer_inputs adirondacks_large
python -m scripts.tools.make_transfer_configs adirondacks_large
```
