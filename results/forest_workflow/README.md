# Forest workflow (same pipeline, different reference layer)

Reference sites come from the forest team's package (`Reference_class.tif`, ~300 m, EPSG:4326:
0 non-forest, 1 forest, 2 Intact Forest Landscape (IFL), 3 IFL with FLII >= 9.6) instead of the
grassland layers. Chosen definition: **class 3 (IFL + high integrity)**; landscapes with any
class-3 area: PACHA, SEKA, Sabah, SOKNOT (none in Carpathian, Central India, Nepal TAL, KAZA).

Within-AOI reference area: PACHA 40,647 km2, SEKA 1,643, Sabah 1,149, SOKNOT 454.

1. `python -m scripts.tools.make_forest_reference seka sabah soknot pacha` (local; writes the 500 m
   reference and forest-extent rasters into `forest_inputs/`, tracked in git, a cell counts when
   >= 50% of its 300 m source pixels are class 3 / forest).
2. `python -m scripts.tools.make_forest_configs seka sabah soknot pacha`
3. `./run_forest_pipeline.sh <name>`: `scripts.tools.fetch_forest_tiles` downloads bands d02-d39
   from Earth Engine and substitutes the local reference raster for band d01 (cheaper than the
   grassland fetch: the multi-year d01 computation is dropped), then sample, fits and map.
   Nothing in the original scripts is modified; the fetcher plugs into `cache_aoi_tiles(tile_fetcher=...)`.

Open points: (a) `Forest_mask` marks 99% of Carpathian and 97% of PACHA as forest, so it looks like
a forest *zone*, not actual forest cover - the inference mask should be confirmed with the forest
team; (b) the stack still carries grassland-oriented variables (e.g. short vegetation height).

## First results (SEKA and Sabah, reference = IFL + FLII >= 9.6)
- Reference area: SEKA 1,643 km2 (10% of its forest AOI), Sabah 1,149 km2 (1.6%).
- Predictors-only held-out R2 (mean): SEKA -0.22, Sabah -0.16 (only 1 response > 0.3 in each);
  all-variables 0.61 / 0.49 (inflated). SEKA uses 75 km blocks / 4 folds (only 4 reference blocks).
- Departure maps are not interpretable as integrity: 54% (SEKA) and 75% (Sabah) of scored forest is
  beyond the 95th percentile. Only 26% (SEKA) / 32% (Sabah) of forest lies inside the reference
  environmental envelope: the IFL reference sits in cooler (-1 sd), more rugged (+1.0-1.2 sd
  topographic diversity) terrain than most of the forest, so scores extrapolate.
- Reference responses are nearly saturated (e.g. SEKA max LAI sd 0.15 vs 0.70 in other forest),
  leaving little variance for the environmental models to explain.
