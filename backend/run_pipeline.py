"""STEP 7 (extended by Upgrade v2) - run the whole JalDrishti pipeline.

Order:
  1. fetch_rainfall.py         (live Open-Meteo, SIMULATED fixture fallback)
  2. fetch_discharge.py        (live GloFAS via Open-Meteo Flood API, fallback)
  3. fetch_cwc_aff.py          (CWC official gauges, AFF public feed; 3-hourly cap;
                                capped-cadence public files)
  4. fetch_villages.py         (live Overpass coords, approximate fallback)
  5. fetch_archive.py          (archived Jun-Aug 2026 window; skips if present)
  6. classify_risk.py          (demo heuristic -> public payloads)
  7. classify_risk.py --replay (same heuristic over the 2026-07 archive)
  8. analyze_replay.py         (REPLAY-FINDINGS.md + findings card JSON)
  9. fetch_s1_footprints.py    (CDSE STAC: 2022 Silchar + 2026 Upper Assam)
 10. s1_water_extent.py        (openEO water extent; degrades honestly
                                without CDSE credentials, never fabricates)
 11. check_provenance.py       (gate: every value has source/retrieved_at/class)
 12. check_plain_language.py   (gate: no technical jargon in the viewer's
                                default view - Step E acceptance)

Each step runs as its own process; the pipeline stops at the first
non-zero exit and propagates it. Exit 0 means: payloads regenerated AND
the provenance gate passed. The individual scripts own the degraded-mode
(SIMULATED fixture) behaviour, so a live-API outage does NOT fail the
pipeline - it produces labelled fixtures instead.

Run from anywhere:  python backend/run_pipeline.py
Then serve the map: python -m http.server 8000 --directory public
"""

import os
import pathlib
import subprocess
import sys
import time

# Optional per-step watchdog (seconds). Unset locally; the cloud workflow
# sets it so a hanging fetch becomes a NAMED failure at the exact step
# instead of an anonymous 45-minute stall.
STEP_TIMEOUT = int(os.environ.get("PIPELINE_STEP_TIMEOUT", "0")) or None

BACKEND = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND / "model"))
from basins import basin_ids  # noqa: E402

STEPS = [
    ("rainfall (Open-Meteo, live)", "fetch_rainfall.py", []),
    ("discharge (GloFAS, live)", "fetch_discharge.py", []),
    ("CWC official gauges (AFF public feed, 3-hourly cap)", "fetch_cwc_aff.py", []),
    ("official alerts (NDMA SACHET public feed, 3-hourly cap)", "fetch_sachet.py", []),
    ("GloFAS forecast archive (per-run issue dates)", "archive_glofas_fc.py", []),
    ("forecast scoreboard (ready-made forecasts vs observed)", "model/scoreboard.py", []),
    ("villages (Overpass coords, live)", "fetch_villages.py", []),
    ("archived 2026-07 window (Open-Meteo archive + Flood API)", "fetch_archive.py", []),
    *[(f"model v0 inference ({b}; degrades to heuristic)", "model/predict.py", ["--basin", b])
      for b in basin_ids()],
    ("model forecast ledger (prediction archive)", "archive_model_fc.py", []),
    *[(f"input drift monitor ({b})", "model/drift.py", ["--basin", b]) for b in basin_ids()],
    ("risk classification (model v0 for Dikhow, heuristic elsewhere)", "classify_risk.py", []),
    ("replay of July 2026 event (same heuristic, archived inputs)",
     "classify_risk.py", ["--replay", "2026-07-10", "2026-08-05"]),
    *[(f"model v0 out-of-sample test 2026 ({b})", "model/test_2026.py", ["--basin", b])
      for b in basin_ids()],
    ("replay analysis -> REPLAY-FINDINGS.md (incl. model section)", "analyze_replay.py", []),
    ("Sentinel-1 footprints (CDSE STAC, 2022 + 2026 events)", "fetch_s1_footprints.py", []),
    ("Sentinel-1 water extent (openEO; degrades without CDSE creds)", "s1_water_extent.py", []),
    ("provenance gate", "check_provenance.py", []),
    ("plain-language gate (default view)", "check_plain_language.py", []),
]


def main() -> int:
    t_all = time.monotonic()
    for i, (label, script, args) in enumerate(STEPS, 1):
        if script.endswith("predict.py"):
            time.sleep(7)   # space out Open-Meteo flood-API calls (429 insurance)
        print(f"\n[{i}/{len(STEPS)}] {label} - {script} {' '.join(args)}")
        t = time.monotonic()
        try:
            rc = subprocess.call([sys.executable, str(BACKEND / script), *args],
                                 cwd=BACKEND, timeout=STEP_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"    -> STEP TIMED OUT after {STEP_TIMEOUT}s")
            print(f"\nPIPELINE FAILED at {script} (watchdog timeout)")
            return 1
        print(f"    -> exit {rc} in {time.monotonic() - t:.1f}s")
        if rc != 0:
            print(f"\nPIPELINE FAILED at {script} (exit {rc})")
            return rc
    print(f"\nPIPELINE OK in {time.monotonic() - t_all:.1f}s - payloads in public/, "
          f"serve with: python -m http.server 8000 --directory public")
    return 0


if __name__ == "__main__":
    sys.exit(main())
