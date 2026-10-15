"""Run the whole pipeline for every landscape, unattended.

Usage:
    python -m scripts.tools.run_all_landscapes [--parallel 2] [--only NAME ...]

For each landscape: build the AOI and configs, choose reference thresholds with
the Earth Engine sweep (strictest triple with enough reference pixels), run the
pipeline (``run_landscape_pipeline.sh``), and if the model fit fails, retry with the
next looser threshold triple (up to ``MAX_FALLBACKS``). Failures never stop the
other landscapes. Writes ``logs/all_landscapes_summary.json`` and ``.md``.
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .reference_threshold_sweep import LADDER

ROOT = Path(__file__).resolve().parents[2]
WWF = Path.home() / "Desktop" / "ConnectivityWork" / "WWF_shapefiles"
EE_PROJECT = "charged-sled-403915"
MAX_FALLBACKS = 2
PY = str(ROOT / ".venv" / "bin" / "python")

# name, display name, source files (relative to WWF_shapefiles), AOI mode, buffer km,
# min part km2. "dissolve" = a landscape boundary file; "hull" = convex hull of
# corridors/PAs when no boundary file exists.
LANDSCAPES = [
    ("pacha", "PACHA", [
        "PACHA corridor prioritization/extracted/PACHA_corridors_with_scores.shp",
        "PACHA corridor prioritization/extracted_core/prioritization_core_areas.shp",
    ], "hull", 25, 0),
    ("kaza", "KAZA", ["kaza/kaza_boundary_2014.shp"], "dissolve", 0, 0),
    ("carpathian", "Carpathian",
     ["Carpathian/Background data/DATA_podklady/carpathians_border.shp"],
     "dissolve", 0, 0),
    ("soknot", "SOKNOT",
     ["SOKNOT CORRIDOR NETWORK and Geoscope/SOKNOT BOUNDARY.shp"],
     "dissolve", 0, 0),
    ("sabah", "Sabah", ["Sabah_extracted/sabah_boundary/sabah boundary.shp"],
     "dissolve", 0, 5),
    ("nepal_tal", "Nepal TAL", ["Nepal polygons/talb_dd.shp"], "dissolve", 0, 0),
    ("seka", "SEKA", [
        "Koridor SEKA/Ketentuan_Khusus_Migrasi_Satwa.shp",
        "Koridor SEKA/RB_164.shp",
    ], "hull", 25, 0),
]


def run(command, log_path):
    with open(log_path, "a") as log:
        log.write("$ " + " ".join(map(str, command)) + "\n")
        log.flush()
        return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def make_configs(name, display, sources, mode, buffer_km, min_part, thresholds, log):
    command = [PY, "-m", "scripts.tools.make_landscape_config", name, display,
               *[str(WWF / s) for s in sources], "--ee-project", EE_PROJECT,
               "--aoi-mode", mode, "--buffer-km", str(buffer_km),
               "--min-part-km2", str(min_part)]
    if thresholds:
        command += ["--reference-thresholds", *map(str, thresholds)]
    return run(command, log) == 0


DEFAULT_MIN_PIXELS = 3000
# Marginal landscapes: accept the loosest triple if it still has ~1,000 pixels.
MIN_PIXELS = {"sabah": 1000, "nepal_tal": 1000}


def choose_threshold_index(sweep_log, min_pixels):
    """Strictest ladder index whose logged 500 m reference-pixel count is enough."""
    counts = {}
    for line in sweep_log.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit():
            try:
                counts[(float(parts[0]), float(parts[1]), float(parts[2]))] = float(parts[3])
            except ValueError:
                pass
    for index, triple in enumerate(LADDER):
        if counts.get(tuple(map(float, triple)), 0) >= min_pixels:
            return index
    return None


def process(entry):
    name, display, sources, mode, buffer_km, min_part = entry
    logs = ROOT / "logs" / name
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / "orchestrator.log"
    record = {"landscape": name, "status": "failed", "attempts": []}
    started = time.time()
    if not make_configs(name, display, sources, mode, buffer_km, min_part, None, log):
        record["error"] = "AOI/config creation failed"
        return record
    sweep_log = logs / "sweep.log"
    if not (sweep_log.exists() and "CHOSEN" in sweep_log.read_text()):
        sweep_log.write_text("")
        run([PY, "-m", "scripts.tools.reference_threshold_sweep",
             f"config/{name}_all.toml", "--min-pixels", "1e18"], sweep_log)
    index = choose_threshold_index(sweep_log, MIN_PIXELS.get(name, DEFAULT_MIN_PIXELS))
    if index is None:
        record["error"] = "no threshold triple gave enough reference pixels"
        return record
    for attempt in range(MAX_FALLBACKS + 1):
        if index >= len(LADDER):
            break
        triple = LADDER[index]
        make_configs(name, display, sources, mode, buffer_km, min_part, triple, log)
        (logs / "DONE").unlink(missing_ok=True)
        (logs / "steps.log").unlink(missing_ok=True)
        code = run(["bash", "run_landscape_pipeline.sh", name], log)
        steps = (logs / "steps.log").read_text() if (logs / "steps.log").exists() else ""
        record["attempts"].append({"thresholds": triple, "exit_code": code,
                                   "failed_step": steps.strip().splitlines()[-1]
                                   if "FAILED" in steps else None})
        if code == 0 and (logs / "DONE").exists():
            record["status"] = "ok"
            record["thresholds"] = triple
            break
        failed_step = record["attempts"][-1]["failed_step"] or ""
        if "fit_pred" not in failed_step:
            break  # only a looser reference definition can fix a model-fit failure
        index += 1
    record["hours"] = round((time.time() - started) / 3600, 2)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    entries = [e for e in LANDSCAPES if not args.only or e[0] in args.only]
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for record in pool.map(process, entries):
            results.append(record)
            summary = ROOT / "logs" / "all_landscapes_summary.json"
            summary.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
