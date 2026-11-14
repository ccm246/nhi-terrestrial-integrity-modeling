#!/bin/bash
# Usage: ./run_split_pipeline.sh NAME KIND [SOURCE_CACHE]     KIND = forestv2 or grassv2
#   optional SOURCE_CACHE forces the tile source, e.g. data/gee_cache_guatemala_forestv2
#   e.g. ./run_split_pipeline.sh soknot grassv2
# Habitat-split run: reference sites and inference mask come from the forest/grassland habitat split
# (forest_inputs/forest_reference_v2 + forest_masks_v2, or grass_inputs/grass_reference_v2 + grass_masks_v2).
# Tiles: if the landscape's original cache (data/gee_cache_NAME, or data/gee_raster_cache for
# central_india_boundary) has tiles, they are cloned with band d01 swapped (no Earth Engine download);
# otherwise they are fetched. Then sample -> predictors-only fit -> all-variables fit -> integrity map.
set -u
ulimit -n 4096 2>/dev/null || ulimit -n 2048 2>/dev/null || true
NAME=$1; KIND=$2; SRC_ARG=${3:-}
cd "$(dirname "$0")"
PY=.venv/bin/python
RUN=${NAME}_${KIND}
L=logs/${RUN}; mkdir -p "$L"; : > "$L/steps.log"; rm -f "$L/DONE"
ALL=config/${RUN}_all.toml
PRED=config/${RUN}_predictors_only.toml
if [ "$KIND" = forestv2 ]; then REF=forest_inputs/forest_reference_v2/${NAME}_reference_500m.tif
else REF=grass_inputs/grass_reference_v2/${NAME}_reference_500m.tif; fi
# Tile source: SOURCE_CACHE if given; else the landscape's original cache, else a finished sibling run's cache (same tiles, other reference band).
SRC="$SRC_ARG"
for c in data/gee_cache_${NAME} data/gee_raster_cache data/gee_cache_${NAME}_forestv2 data/gee_cache_${NAME}_grassv2; do
  [ "$c" = data/gee_raster_cache ] && [ "$NAME" != central_india_boundary ] && continue
  [ -n "$SRC" ] && break
  [ "$c" = data/gee_cache_${RUN} ] && continue
  if [ "$(ls $c/tiles/*/*.tif 2>/dev/null | wc -l)" -gt 0 ]; then SRC=$c; break; fi
done
S=outputs/samples/${RUN}_all_2018_spatial_sample.parquet
step() { local log=$1; shift; echo "$(date +%T) $*" >> "$L/steps.log"; "$@" > "$L/$log.log" 2>&1 || { echo "FAILED: $log" >> "$L/steps.log"; return 1; }; }
run() { step "$@" || exit 1; }
cloned=0
if [ -n "$SRC" ]; then
  step clone $PY -m scripts.tools.clone_cache_with_reference $ALL $SRC $REF && cloned=1
  [ $cloned = 0 ] && sed -i '' '/^FAILED: clone/d' "$L/steps.log"
fi
if [ $cloned = 0 ]; then
  for try in 1 2 3 4 5 6; do
    step fetch_$try $PY -m scripts.tools.fetch_forest_tiles $ALL --reference $REF --no-progress && break
    [ $try = 6 ] && { echo "FAILED: fetch" >> "$L/steps.log"; exit 1; }
    sed -i '' '/^FAILED: fetch_/d' "$L/steps.log"; sleep 120
  done
fi
run sample   $PY -m scripts.build_spatial_sample $ALL --no-progress
run fit_pred $PY -m scripts.fit_grassland_integrity_parameters $PRED $S \
      --output-directory outputs/integrity_parameters/${RUN}_predictors_only --no-progress
run fit_all  $PY -m scripts.fit_all_variables_models $ALL $S \
      --output-directory outputs/integrity_parameters/${RUN}_all_variables --no-progress
run apply    $PY -m scripts.apply_reference_condition_models $PRED \
      outputs/integrity_parameters/${RUN}_predictors_only --no-progress
echo done > "$L/DONE"
