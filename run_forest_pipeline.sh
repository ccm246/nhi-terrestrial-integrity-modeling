#!/bin/bash
# Usage: ./run_forest_pipeline.sh LANDSCAPE_NAME
# Forest workflow: tile fetch (Earth Engine bands d02-d39 + local reference band d01 from
# forest_inputs/forest_reference) -> sample -> predictors-only fit -> all-variables fit ->
# integrity map masked to forest_inputs/forest_masks. Configs: config/<name>_forest_*.toml.
set -u
ulimit -n 4096 2>/dev/null || ulimit -n 2048 2>/dev/null || true
NAME=$1
cd "$(dirname "$0")"
PY=.venv/bin/python
L=logs/${NAME}_forest; mkdir -p "$L"; : > "$L/steps.log"; rm -f "$L/DONE"
ALL=config/${NAME}_forest_all.toml
PRED=config/${NAME}_forest_predictors_only.toml
S=outputs/samples/${NAME}_forest_all_2018_spatial_sample.parquet
step() { local log=$1; shift; echo "$(date +%T) $*" >> "$L/steps.log"; "$@" > "$L/$log.log" 2>&1 || { echo "FAILED: $log" >> "$L/steps.log"; return 1; }; }
run() { step "$@" || exit 1; }
for try in 1 2 3 4 5 6; do
  step fetch_$try $PY -m scripts.tools.fetch_forest_tiles $ALL --no-progress && break
  [ $try = 6 ] && { echo "FAILED: fetch" >> "$L/steps.log"; exit 1; }
  sed -i '' '/^FAILED: fetch_/d' "$L/steps.log"; sleep 120
done
run sample   $PY -m scripts.build_spatial_sample $ALL --no-progress
run fit_pred $PY -m scripts.fit_grassland_integrity_parameters $PRED $S \
      --output-directory outputs/integrity_parameters/${NAME}_forest_predictors_only --no-progress
run fit_all  $PY -m scripts.fit_all_variables_models $ALL $S \
      --output-directory outputs/integrity_parameters/${NAME}_forest_all_variables --no-progress
run apply    $PY -m scripts.apply_reference_condition_models $PRED \
      outputs/integrity_parameters/${NAME}_forest_predictors_only --no-progress
echo done > "$L/DONE"
