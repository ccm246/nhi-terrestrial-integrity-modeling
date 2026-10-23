# Setup notes for running this pipeline on a new machine

Problems hit on a second Mac, and the fixes (all verified):

1. **Python / packages**: Python 3.12 via `uv venv --python 3.12 .venv`, then
   `uv pip install -p .venv -r requirements.txt geopandas "pandas<3"`.
2. **pandas must be < 3.** With pandas 3.x the fit step crashes on large samples
   (e.g. KAZA, ~117k rows) with `ValueError: output array is read-only` in
   `create_observed_expected_figure` (`probabilities /= probabilities.sum()`).
   pandas 2.3.3 works.
3. **rasterio on macOS < 15**: the rasterio 1.5.2 Apple-Silicon wheel needs macOS 15,
   so uv compiles it against a system GDAL whose bundled Arrow clashes with pyarrow
   (`ArrowKeyError: ... scheme 'file' ... already registered`) when writing Parquet.
   Install a prebuilt wheel instead: `uv pip install --reinstall --only-binary rasterio "rasterio==1.4.1"`.
4. **Open-file limit**: macOS allows 256 open files per process, which breaks the
   grassland mask on large AOIs (fixed in code) and the inference worker pool
   (`Too many open files` -> `BrokenProcessPool`). `run_landscape_pipeline.sh` raises
   the limit; when running `apply_reference_condition_models` by hand run
   `ulimit -n 4096` first in the same terminal.
5. **Earth Engine**: tile downloads need a project with quota left
   (`scripts.tools.prepare_gee_downloads --ee-project YOUR_PROJECT`).
