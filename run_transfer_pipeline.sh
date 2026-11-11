#!/bin/bash
# Usage: ./run_transfer_pipeline.sh NAME REF [SOURCE_REF]      e.g. ./run_transfer_pipeline.sh adirondacks_large hm
#                                                                   ./run_transfer_pipeline.sh adirondacks_large wild hm
# Adirondack transfer workflow: models trained on reference sites (REF = hm or wild) inside the
# Adirondack Park, applied to the whole forest zone of the large extent. Tile fetch (Earth Engine bands
# d02-d39 + local reference band d01 from transfer_inputs/NAME/reference_REF_500m.tif) -> sample ->
# predictors-only fit -> all-variables fit -> integrity map. With SOURCE_REF, the tiles are cloned from
# that run's finished cache with the reference band swapped (no second download).
set -u
ulimit -n 4096 2>/dev/null || ulimit -n 2048 2>/dev/null || true
NAME=$1; REF=$2; SRC=${3:-}
cd "$(dirname "$0")"
PY=.venv/bin/python
RUN=${NAME}_${REF}
L=logs/${RUN}; mkdir -p "$L"; : > "$L/steps.log"; rm -f "$L/DONE"
ALL=config/${RUN}_all.toml
PRED=config/${RUN}_predictors_only.toml
REFTIF=transfer_inputs/${NAME}/reference_${REF}_500m.tif
S=outputs/samples/${RUN}_all_2018_spatial_sample.parquet
step() { local log=$1; shift; echo "$(date +%T) $*" >> "$L/steps.log"; "$@" > "$L/$log.log" 2>&1 || { echo "FAILED: $log" >> "$L/steps.log"; return 1; }; }
run() { step "$@" || exit 1; }
if [ -n "$SRC" ]; then
  run clone $PY -m scripts.tools.clone_cache_with_reference $ALL data/gee_cache_${NAME}_${SRC} $REFTIF
else
  for try in 1 2 3 4 5 6; do
    step fetch_$try $PY -m scripts.tools.fetch_forest_tiles $ALL --reference $REFTIF --no-progress && break
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
