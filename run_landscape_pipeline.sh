#!/bin/bash
# Usage: ./run_landscape_pipeline.sh LANDSCAPE_NAME
# Runs the full pipeline for a landscape whose configs were made by
# scripts.tools.make_landscape_config: tile fetch -> grassland mask -> sample ->
# predictors-only fit -> all-variables fit -> integrity map (predictors-only).
set -u
# macOS allows only 256 open files per process by default; the inference worker pool and
# model saving exceed that on larger landscapes ("Too many open files").
ulimit -n 4096 2>/dev/null || ulimit -n 2048 2>/dev/null || true
NAME=$1
cd "$(dirname "$0")"
PY=.venv/bin/python
L=logs/$NAME; mkdir -p "$L"
ALL=config/${NAME}_all.toml
PRED=config/${NAME}_predictors_only.toml
S=outputs/samples/${NAME}_all_2018_spatial_sample.parquet
step() { local log=$1; shift; echo "$(date +%T) $*" >> "$L/steps.log"; "$@" > "$L/$log.log" 2>&1 || { echo "FAILED: $log" >> "$L/steps.log"; return 1; }; }
run() { step "$@" || exit 1; }
# Fetch resumes from cached tiles, so retry through transient Earth Engine errors.
for try in 1 2 3 4 5 6; do
  step fetch_$try $PY -m scripts.fetch_gee_raster_tiles $ALL --no-progress && break
  [ $try = 6 ] && { echo "FAILED: fetch" >> "$L/steps.log"; exit 1; }
  sed -i '' '/^FAILED: fetch_/d' "$L/steps.log"; sleep 120
done
# The mask only needs Earth Engine if it does not exist yet (e.g. not unzipped from a hand-off).
[ -f data/grassland_masks/${NAME}_all_grassland_mask_2018.tif ] || \
run mask         $PY -m scripts.tools.make_grassland_mask $ALL
run sample       $PY -m scripts.build_spatial_sample $ALL --no-progress
run fit_pred     $PY -m scripts.fit_grassland_integrity_parameters $PRED $S \
                    --output-directory outputs/integrity_parameters/${NAME}_predictors_only --no-progress
run fit_all      $PY -m scripts.fit_all_variables_models $ALL $S \
                    --output-directory outputs/integrity_parameters/${NAME}_all_variables --no-progress
run apply        $PY -m scripts.apply_reference_condition_models $PRED \
                    outputs/integrity_parameters/${NAME}_predictors_only --no-progress
echo done > "$L/DONE"
