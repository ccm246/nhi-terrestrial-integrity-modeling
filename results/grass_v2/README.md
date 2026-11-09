# Grass v2: habitat-split grassland zone and reference (second-machine instructions)

Grassland zone = habitat split code 3 (natural, Hansen canopy <= 20 %) plus, with `--dry-woodland`,
code 4 (ESA tree cover or mangrove whose canopy fails the 20 % test: open dry woodland), instead of the
Global Pasture Watch >= 80 % mask. Reference sites = grassland zone AND human modification <= 0.1 AND
human influence < 0.08 (same ladder as forest v2; strictest rung reached the minimum).
Responses and cross-validation settings are those of the original grassland run (config/kaza_*.toml).

KAZA: grassland zone 469,750 km2 (old mask scored 29,146 km2 + 45,815 km2 reference), reference 250,188 km2.
Only use `--dry-woodland` where code 4 is open woodland (KAZA, SOKNOT-type savanna), not plantations
or regrowing forest (Sabah, SEKA).

## Steps (in the repo folder on the second machine)

```bash
git stash            # only if git complains about local changes
git pull origin central-india-variants
source .venv/bin/activate
ulimit -n 4096
./run_grass_v2_pipeline.sh kaza
```

The run writes logs/kaza_grassv2/steps.log and, when finished, a DONE file in the same folder.
KAZA is large: expect several hours (tile fetch from Earth Engine, then fits).

## Results to send back

```bash
zip -r grassv2_results.zip outputs/integrity_parameters outputs/reference_condition_inference outputs/samples -i "*grassv2*"
```

## Rebuilding the inputs from scratch (this Mac only; needs data/habitat)

```bash
python -m scripts.tools.make_grass_habitat_inputs kaza --dry-woodland
python -m scripts.tools.make_grass_v2_configs kaza
```
