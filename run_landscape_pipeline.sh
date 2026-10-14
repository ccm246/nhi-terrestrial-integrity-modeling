#!/bin/bash
# Usage: ./run_landscape_pipeline.sh LANDSCAPE_NAME
# Runs the full pipeline for a landscape whose configs were made by
# scripts.tools.make_landscape_config: tile fetch -> grassland mask -> sample ->
# predictors-only fit -> all-variables fit -> integrity map (predictors-only).
set -u
NAME=$1
cd "$(dirname "$0")"
PY=.venv/bin/python
L=logs/$NAME; mkdir -p "$L"
ALL=config/${NAME}_all.toml
PRED=config/${NAME}_predictors_only.toml
S=outputs/samples/${NAME}_all_2018_spatial_sample.parquet
step() { local log=$1; shift; echo "$(date +%T) $*" >> "$L/steps.log"; "$@" > "$L/$log.log" 2>&1 || { echo "FAILED: $log" >> "$L/steps.log"; exit 1; }; }
step fetch        $PY -m scripts.fetch_gee_raster_tiles $ALL --no-progress
step mask         $PY -m scripts.tools.make_grassland_mask $ALL
step sample       $PY -m scripts.build_spatial_sample $ALL --no-progress
step fit_pred     $PY -m scripts.fit_grassland_integrity_parameters $PRED $S \
                    --output-directory outputs/integrity_parameters/${NAME}_predictors_only --no-progress
step fit_all      $PY -m scripts.fit_all_variables_models $ALL $S \
                    --output-directory outputs/integrity_parameters/${NAME}_all_variables --no-progress
step apply        $PY -m scripts.apply_reference_condition_models $PRED \
                    outputs/integrity_parameters/${NAME}_predictors_only --no-progress
echo done > "$L/DONE"
