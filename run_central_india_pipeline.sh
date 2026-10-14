#!/bin/bash
# Waits for the tile fetch, then builds the sample and runs both model variants.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
while pgrep -f "scripts.fetch_gee_raster_tiles config/central_india_all.toml" >/dev/null; do sleep 20; done
$PY -m scripts.fetch_gee_raster_tiles config/central_india_all.toml --no-progress > logs/central_india_fetch_verify.log 2>&1 || exit 1
$PY -m scripts.build_spatial_sample config/central_india_all.toml --no-progress > logs/central_india_sample_t30.log 2>&1 || exit 2
S=outputs/samples/central_india_all_2018_spatial_sample.parquet
$PY -m scripts.fit_grassland_integrity_parameters config/central_india_predictors_only.toml $S \
  --output-directory outputs/integrity_parameters/central_india_predictors_only --no-progress > logs/central_india_fit_predictors_only.log 2>&1
$PY -m scripts.fit_all_variables_models config/central_india_all.toml $S \
  --output-directory outputs/integrity_parameters/central_india_all_variables --no-progress > logs/central_india_fit_all_variables.log 2>&1
echo done > logs/central_india_pipeline.done
