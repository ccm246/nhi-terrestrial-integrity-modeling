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
