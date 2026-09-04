"""UPGRADE v2, STEP A1 - fetch ARCHIVED hourly rainfall + daily discharge
around the 19 July 2026 Mon-district cloudburst / Upper Assam flood.

Sources (both free, no key):
- Open-Meteo Historical Weather (Archive) API, archive-api.open-meteo.com -
  hourly precipitation, ERA5-family reanalysis (model analysis, NOT gauges).
- Open-Meteo Flood API (GloFAS v4) with start_date/end_date - VERIFIED to
  support explicit date windows (rules fixed before training

Window fetched: 2026-06-10 .. 2026-08-05. Wider than the replay window
(10 Jul - 5 Aug) because the unchanged heuristic needs a trailing 30-day
discharge window and the analysis needs a trailing-month rainfall baseline.

Grid points: the 13 Step-1 points PLUS Mon town, Nagaland (~26.75, 95.05) -
the cloudburst district; the existing Nagaland points sit 50-90 km away.
River points: the 8 Step-2 points at their already-snapped GloFAS cells.

Everything here is a retrieved reanalysis of the past -> class OBSERVED,
with source strings naming the exact product honestly.

Outputs:
  data/archive/2026-07/raw/*.json            raw API responses
  data/archive/2026-07/archive_rainfall.json
  data/archive/2026-07/archive_discharge.json

Idempotent: if the outputs already exist and cover the window, the fetch is
skipped (archive data is static history). Force with --refresh.

Fallback (per standing rules): a point failing after 3 retries with backoff
gets its attempted request saved under data/fixtures/ and is recorded in
degraded_points. Archived history is NEVER simulated - a degraded point is
simply absent, and downstream steps work with what exists.
"""

import datetime
import sys

from common import (DATA_DIR, OBSERVED, fetch_json, load_json,
                    save_failed_request, save_json, utc_now_iso)
from fetch_rainfall import GRID_POINTS
from fetch_discharge import RIVER_POINTS

ARCHIVE_TAG = "2026-07"
ARCHIVE_DIR = DATA_DIR / "archive" / ARCHIVE_TAG
RAW_ARCH_DIR = ARCHIVE_DIR / "raw"

FETCH_START = "2026-06-10"
FETCH_END = "2026-08-05"
N_DAYS = 57          # 2026-06-10 .. 2026-08-05 inclusive
N_HOURS = N_DAYS * 24

RAIN_API = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"

SRC_RAIN = ("Open-Meteo Historical Weather (Archive) API, ERA5-family "
            "reanalysis (model analysis of past weather, NOT gauge data)")
SRC_DISCH = ("GloFAS v4 reanalysis via Open-Meteo Flood API "
             "start_date/end_date archive window (model reanalysis, "
             "NOT a gauge reading)")

# The cloudburst district. Coordinates are Mon town; the value is for the
# reanalysis grid cell containing it.
MON_POINT = {"id": "mon", "name": "Mon (Nagaland)", "lat": 26.75, "lon": 95.05,
             "region": "Nagaland hills (Mon district)",
             "catchment": "south-bank tributary headwaters (Disang/Dikhow)"}

ARCHIVE_RAIN_POINTS = GRID_POINTS + [MON_POINT]


def outputs_complete() -> bool:
    """True if both archive files exist, cover the window, and include Mon."""
    try:
        rain = load_json(ARCHIVE_DIR / "archive_rainfall.json")
        disch = load_json(ARCHIVE_DIR / "archive_discharge.json")
    except (FileNotFoundError, ValueError):
        return False
    if rain.get("window_start") != FETCH_START or rain.get("window_end") != FETCH_END:
        return False
    rain_ids = {p["id"] for p in rain.get("points", [])}
    disch_ids = {p["id"] for p in disch.get("points", [])}
    return ("mon" in rain_ids and len(rain_ids) >= len(ARCHIVE_RAIN_POINTS) - 2
            and len(disch_ids) >= len(RIVER_POINTS) - 2)


def river_cells() -> list[dict]:
    """The 8 river points at their snapped GloFAS cells (from data/discharge.json)."""
    cells = {}
    try:
        live = load_json(DATA_DIR / "discharge.json")
        cells = {p["id"]: (p.get("grid_lat"), p.get("grid_lon"))
                 for p in live.get("points", [])}
    except (FileNotFoundError, ValueError):
        pass
    out = []
    for p in RIVER_POINTS:
        glat, glon = cells.get(p["id"], (None, None))
        if glat is None:
            from fetch_discharge import snap_to_river_cell
            try:
                glat, glon, _ = snap_to_river_cell(p)
            except RuntimeError:
                glat, glon = p["lat"], p["lon"]
        out.append({**p, "grid_lat": glat, "grid_lon": glon})
    return out


def fetch_rain(retrieved_at: str) -> tuple[list[dict], list[str]]:
    points_out, degraded = [], []
    for p in ARCHIVE_RAIN_POINTS:
        params = {"latitude": p["lat"], "longitude": p["lon"],
                  "start_date": FETCH_START, "end_date": FETCH_END,
                  "hourly": "precipitation", "timezone": "UTC"}
        try:
            api = fetch_json(RAIN_API, params)
            save_json(RAW_ARCH_DIR / f"rainfall_{p['id']}.json", api)
        except RuntimeError as err:
            save_failed_request(f"archive_rainfall_{p['id']}", RAIN_API, params, str(err))
            degraded.append(p["id"])
            print(f"  rain {p['id']:12s} DEGRADED: {err}")
            continue
        hours = []
        for t_str, mm in zip(api["hourly"]["time"], api["hourly"]["precipitation"]):
            if mm is None:
                continue
            hours.append({
                "time": t_str + "Z" if not t_str.endswith("Z") else t_str,
                "precipitation_mm": mm,
                "class": OBSERVED,
                "source": SRC_RAIN,
                "retrieved_at": retrieved_at,
            })
        points_out.append({**p, "api_elevation_m": api.get("elevation"),
                           "retrieved_at": retrieved_at, "hourly": hours})
        print(f"  rain {p['id']:12s} hours={len(hours)}/{N_HOURS}")
    return points_out, degraded


def fetch_discharge(retrieved_at: str) -> tuple[list[dict], list[str]]:
    points_out, degraded = [], []
    for p in river_cells():
        params = {"latitude": p["grid_lat"], "longitude": p["grid_lon"],
                  "start_date": FETCH_START, "end_date": FETCH_END,
                  "daily": "river_discharge", "timezone": "UTC"}
        try:
            api = fetch_json(FLOOD_API, params)
            save_json(RAW_ARCH_DIR / f"discharge_{p['id']}.json", api)
        except RuntimeError as err:
            save_failed_request(f"archive_discharge_{p['id']}", FLOOD_API, params, str(err))
            degraded.append(p["id"])
            print(f"  disch {p['id']:22s} DEGRADED: {err}")
            continue
        days = []
        for d_str, q in zip(api["daily"]["time"], api["daily"]["river_discharge"]):
            if q is None:
                continue
            days.append({
                "date": d_str,
                "discharge_m3s": q,
                "class": OBSERVED,
                "source": SRC_DISCH,
                "retrieved_at": retrieved_at,
            })
        points_out.append({**p, "retrieved_at": retrieved_at, "daily": days})
        print(f"  disch {p['id']:22s} days={len(days)}/{N_DAYS}")
    return points_out, degraded


def main() -> int:
    if "--refresh" not in sys.argv and outputs_complete():
        print(f"OK archive {ARCHIVE_TAG} already present and complete "
              f"(static history; --refresh to re-fetch)")
        return verify()

    retrieved_at = utc_now_iso()
    rain_points, rain_degraded = fetch_rain(retrieved_at)
    disch_points, disch_degraded = fetch_discharge(retrieved_at)

    save_json(ARCHIVE_DIR / "archive_rainfall.json", {
        "generated_at": retrieved_at,
        "source": SRC_RAIN,
        "window_start": FETCH_START, "window_end": FETCH_END,
        "note": ("Archived hourly precipitation around the 19 July 2026 Mon-district "
                 "cloudburst and the Upper Assam floods that followed. Fetch window is "
                 "wider than the 10 Jul - 5 Aug replay window to provide trailing "
                 "baselines. Values are reanalysis for the model grid cell containing "
                 "each named point; no gauge data is used or claimed."),
        "degraded_points": rain_degraded,
        "points": rain_points,
    })
    save_json(ARCHIVE_DIR / "archive_discharge.json", {
        "generated_at": retrieved_at,
        "source": SRC_DISCH,
        "window_start": FETCH_START, "window_end": FETCH_END,
        "note": ("Archived daily GloFAS discharge at the same snapped river-network "
                 "cells used live (see data/discharge.json snap notes). Model "
                 "reanalysis, NOT official gauge readings or danger levels."),
        "degraded_points": disch_degraded,
        "points": disch_points,
    })
    return verify()


def verify() -> int:
    rain = load_json(ARCHIVE_DIR / "archive_rainfall.json")
    disch = load_json(ARCHIVE_DIR / "archive_discharge.json")
    for doc, kind, series in ((rain, "rain", "hourly"), (disch, "discharge", "daily")):
        assert doc["points"], f"archive {kind}: no points at all"
        for pt in doc["points"]:
            assert pt[series], f"archive {kind} {pt['id']}: empty series"
            for rec in pt[series]:
                assert {"source", "retrieved_at", "class"} <= set(rec), \
                    f"archive {kind} {pt['id']}: provenance missing"
                assert rec["class"] == OBSERVED, \
                    f"archive {kind} {pt['id']}: class must be OBSERVED"
    rain_ids = {p["id"] for p in rain["points"]}
    degraded = rain["degraded_points"] + disch["degraded_points"]
    if "mon" not in rain_ids:
        print("WARNING: Mon point missing (degraded) - replay loses the cloudburst anchor")
    print(f"OK data/archive/{ARCHIVE_TAG}: {len(rain['points'])} rain points "
          f"({N_HOURS} h window), {len(disch['points'])} river points ({N_DAYS} d), "
          f"degraded={degraded or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
