#!/bin/bash
# Usage: ./run_wave2_pipeline.sh LANDSCAPE_NAME [noapply]
# Wave 2: refit with only the responses selected by scripts.tools.make_wave2_configs
# (config/<name>_w2_*.toml) on the existing wave-1 sample, then (unless "noapply") make the
# integrity map. Needs outputs/samples/<name>_all_2018_spatial_sample.parquet and, for the
# map, the cached tiles + grassland mask from wave 1. Outputs go to *_w2_* folders so wave 1
# is untouched.
set -u
ulimit -n 4096 2>/dev/null || ulimit -n 2048 2>/dev/null || true
NAME=$1
cd "$(dirname "$0")"
PY=.venv/bin/python
L=logs/${NAME}_w2; mkdir -p "$L"; : > "$L/steps.log"; rm -f "$L/DONE"
S=outputs/samples/${NAME}_all_2018_spatial_sample.parquet
step() { local log=$1; shift; echo "$(date +%T) $*" >> "$L/steps.log"; "$@" > "$L/$log.log" 2>&1 || { echo "FAILED: $log" >> "$L/steps.log"; exit 1; }; }
step fit_pred $PY -m scripts.fit_grassland_integrity_parameters config/${NAME}_w2_predictors_only.toml $S \
      --output-directory outputs/integrity_parameters/${NAME}_w2_predictors_only --no-progress
step fit_all  $PY -m scripts.fit_all_variables_models config/${NAME}_w2_all.toml $S \
      --output-directory outputs/integrity_parameters/${NAME}_w2_all_variables --no-progress
if [ "${2:-}" != "noapply" ]; then
  step apply  $PY -m scripts.apply_reference_condition_models config/${NAME}_w2_predictors_only.toml \
      outputs/integrity_parameters/${NAME}_w2_predictors_only --no-progress
fi
echo done > "$L/DONE"
