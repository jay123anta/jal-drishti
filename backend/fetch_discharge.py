"""STEP 2 - fetch daily river discharge (REAL, LIVE) from the Open-Meteo
Flood API (GloFAS v4, no key, endpoint flood-api.open-meteo.com).

Window: past 30 days + 30-day forecast, daily, at 8 river points.

Provenance (honest labelling):
- past days come from the GloFAS consolidated reanalysis chain -> class
  OBSERVED, source states clearly this is MODEL REANALYSIS, not a gauge.
- future days are the GloFAS ensemble forecast -> class FORECAST.
- ensemble min/median/max are requested as daily statistics variables and
  attached where the API returns them (typically forecast days only).

Raw responses -> data/raw/discharge_<id>.json
Normalised    -> data/discharge.json
"""

import datetime
import math
import sys
import time

from common import (DATA_DIR, FIXTURES_DIR, FORECAST, OBSERVED, RAW_DIR,
                    SIMULATED, fetch_json, save_failed_request, save_json,
                    utc_now_iso)

API_URL = "https://flood-api.open-meteo.com/v1/flood"

# qmin/qmax: plausible monsoon discharge band (m3/s) used ONLY to pick the
# right river-network grid cell (mainstem vs tributary), not to alter values.
RIVER_POINTS = [
    {"id": "brahmaputra_dibrugarh", "river": "Brahmaputra", "site": "Dibrugarh", "lat": 27.47, "lon": 94.91, "qmin": 4000, "qmax": None},
    {"id": "brahmaputra_tezpur",    "river": "Brahmaputra", "site": "Tezpur",    "lat": 26.63, "lon": 92.79, "qmin": 4000, "qmax": None},
    {"id": "brahmaputra_guwahati",  "river": "Brahmaputra", "site": "Guwahati",  "lat": 26.17, "lon": 91.74, "qmin": 4000, "qmax": None},
    {"id": "barak_silchar",         "river": "Barak",       "site": "Silchar",   "lat": 24.83, "lon": 92.78, "qmin": 300,  "qmax": 8000},
    {"id": "dikhow_sivasagar",      "river": "Dikhow",      "site": "near Sivasagar", "lat": 26.98, "lon": 94.64, "qmin": 50, "qmax": 2500},
    {"id": "kopili_kampur",         "river": "Kopili",      "site": "near Kampur (Nagaon)", "lat": 26.17, "lon": 92.53, "qmin": 100, "qmax": 5000},
    {"id": "jiabharali_tezpur",     "river": "Jia Bharali", "site": "near NH52 crossing (Sonitpur)", "lat": 26.79, "lon": 92.87, "qmin": 200, "qmax": 8000},
    {"id": "beki_barpeta",          "river": "Beki",        "site": "near Barpeta Road", "lat": 26.45, "lon": 90.97, "qmin": 100, "qmax": 8000},
    # Phase 1c (generic coverage): the two remaining July-2026 flood paths, both
    # with CWC flood-forecast stations (GOLAGHAT / NANGLAMORAGHAT)
    {"id": "dhansiri_golaghat",     "river": "Dhansiri (South)", "site": "Golaghat",   "lat": 26.504, "lon": 93.952, "qmin": 50,  "qmax": 4000},
    {"id": "disang_nanglamoraghat", "river": "Disang",      "site": "Nanglamoraghat (Sivasagar)", "lat": 26.985, "lon": 94.777, "qmin": 30, "qmax": 3000},
    # Goal C (2026-09-02): the two largest un-modelled tributaries, both with
    # CWC AFF stations (BADATIGHAT / MANAS N H CROSSING) at these coordinates
    {"id": "subansiri_badatighat",  "river": "Subansiri",   "site": "Badatighat (Lakhimpur)", "lat": 26.949, "lon": 93.962, "qmin": 500, "qmax": None},
    {"id": "manas_nhcrossing",      "river": "Manas",       "site": "NH crossing (Bongaigaon)", "lat": 26.464, "lon": 90.750, "qmin": 100, "qmax": 10000},
    # Goal D (2026-09-04): Ranganadi (dam-release flash floods, Lakhimpur)
    # and Katakhal/Dhaleswari (Mizoram's Tlawng entering Hailakandi) - both
    # with CWC AFF stations at these coordinates
    {"id": "ranganadi_ntxing",      "river": "Ranganadi",   "site": "NT Road crossing (Lakhimpur)", "lat": 27.204, "lon": 94.060, "qmin": 15, "qmax": 3000},
    {"id": "katakhal_matijuri",     "river": "Katakhal",    "site": "Matijuri (Hailakandi)", "lat": 24.649, "lon": 92.609, "qmin": 10, "qmax": 3000},
    # coverage-watch addition (2026-09-04): active SACHET alert on an uncovered river
    {"id": "sankosh_golokganj",     "river": "Sankosh",     "site": "Golokganj (Dhubri)", "lat": 26.107, "lon": 89.824, "qmin": 100, "qmax": None},
]

PARAMS_TEMPLATE = {
    "daily": "river_discharge,river_discharge_mean,river_discharge_median,"
             "river_discharge_max,river_discharge_min",
    "past_days": 30,
    "forecast_days": 30,
    "timezone": "UTC",
}

SRC_PAST = ("GloFAS v4 consolidated reanalysis via Open-Meteo Flood API "
            "(model reanalysis, NOT a gauge reading)")
SRC_FUTURE = "GloFAS v4 ensemble forecast via Open-Meteo Flood API"

# GloFAS runs on a 0.05-deg grid; a named-town coordinate often falls on a
# land cell with near-zero discharge. Standard practice is to snap to the
# river-network cell nearby. We probe a +/-0.15-deg neighbourhood and keep
# the cell with the highest 7-day mean discharge (the main channel).
SNAP_STEP = 0.05
SNAP_RADIUS_CELLS = 3


def snap_to_river_cell(p: dict) -> tuple[float, float, float]:
    """Return (lat, lon, probe_mean_m3s) of the max-discharge cell near p."""
    lats, lons = [], []
    for di in range(-SNAP_RADIUS_CELLS, SNAP_RADIUS_CELLS + 1):
        for dj in range(-SNAP_RADIUS_CELLS, SNAP_RADIUS_CELLS + 1):
            lats.append(round(p["lat"] + di * SNAP_STEP, 4))
            lons.append(round(p["lon"] + dj * SNAP_STEP, 4))
    probe = fetch_json(API_URL, {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "daily": "river_discharge",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "UTC",
    })
    cells = probe if isinstance(probe, list) else [probe]
    qmin, qmax = p.get("qmin") or 0, p.get("qmax")
    qualifying = []   # (dist2, lat, lon, mean) within the plausible band
    fallback = None   # max-mean cell, used only if nothing qualifies
    for cell in cells:
        vals = [v for v in cell.get("daily", {}).get("river_discharge", []) if v is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        lat, lon = cell["latitude"], cell["longitude"]
        if fallback is None or mean > fallback[2]:
            fallback = (lat, lon, mean)
        if mean >= qmin and (qmax is None or mean <= qmax):
            d2 = (lat - p["lat"]) ** 2 + (lon - p["lon"]) ** 2
            qualifying.append((d2, lat, lon, mean))
    if qualifying:
        _, lat, lon, mean = min(qualifying)
        return lat, lon, mean
    if fallback is not None:
        return fallback
    return p["lat"], p["lon"], None


def simulated_days(point: dict, today: datetime.date) -> list[dict]:
    """Deterministic clearly-labelled SIMULATED fallback series."""
    seed = sum(ord(c) for c in point["id"])
    base = 500.0 + (seed % 40) * 100.0
    out = []
    for i in range(-30, 31):
        d = today + datetime.timedelta(days=i)
        q = base * (1.0 + 0.35 * math.sin((i + seed) / 6.0))
        out.append({
            "date": d.isoformat(),
            "discharge_m3s": round(q, 1),
            "class": SIMULATED,
            "source": "SIMULATED fixture (live Open-Meteo Flood API call failed)",
            "retrieved_at": utc_now_iso(),
            "simulated": True,
        })
    return out


def normalise_days(api: dict, retrieved_at: str) -> list[dict]:
    today = datetime.datetime.now(datetime.timezone.utc).date()
    d = api["daily"]
    out = []
    for i, date_str in enumerate(d["time"]):

        def col(name):
            vals = d.get(name)
            return vals[i] if vals and i < len(vals) else None

        q = col("river_discharge")
        if q is None:
            continue
        date = datetime.date.fromisoformat(date_str)
        past = date <= today
        rec = {
            "date": date_str,
            "discharge_m3s": q,
            "class": OBSERVED if past else FORECAST,
            "source": SRC_PAST if past else SRC_FUTURE,
            "retrieved_at": retrieved_at,
        }
        ens = {k: col(f"river_discharge_{k}") for k in ("min", "median", "mean", "max")}
        if any(v is not None for v in ens.values()):
            rec["ensemble"] = {k: v for k, v in ens.items() if v is not None}
            rec["ensemble_note"] = "min/median/mean/max across GloFAS ensemble members"
        out.append(rec)
    return out


def main() -> int:
    points_out = []
    degraded = []
    for p in RIVER_POINTS:
        time.sleep(2)   # space out flood-API calls (rate-limit insurance)
        retrieved_at = utc_now_iso()
        try:
            snap_lat, snap_lon, probe_mean = snap_to_river_cell(p)
        except RuntimeError:
            snap_lat, snap_lon, probe_mean = p["lat"], p["lon"], None
        params = dict(PARAMS_TEMPLATE, latitude=snap_lat, longitude=snap_lon)
        try:
            api = fetch_json(API_URL, params)
            save_json(RAW_DIR / f"discharge_{p['id']}.json", api)
            days = normalise_days(api, retrieved_at)
            live = True
        except RuntimeError as err:
            save_failed_request(f"discharge_{p['id']}", API_URL, params, str(err))
            days = simulated_days(p, datetime.datetime.now(datetime.timezone.utc).date())
            save_json(FIXTURES_DIR / f"discharge_{p['id']}.json",
                      dict(p, simulated=True, daily=days))
            live = False
            degraded.append(p["id"])
        n_obs = sum(1 for r in days if r["class"] == OBSERVED)
        n_fc = sum(1 for r in days if r["class"] == FORECAST)
        n_ens = sum(1 for r in days if "ensemble" in r)
        points_out.append({
            **p,
            "grid_lat": snap_lat, "grid_lon": snap_lon,
            "snap_note": ("snapped to the max-mean-discharge GloFAS cell within "
                          "+/-0.15 deg of the named site (river-network cell selection)"),
            "live": live, "retrieved_at": retrieved_at, "daily": days,
        })
        latest = next((r["discharge_m3s"] for r in reversed(days) if r["class"] == OBSERVED), None)
        print(f"  {p['id']:24s} live={live} grid=({snap_lat},{snap_lon}) "
              f"latest={latest} m3/s past={n_obs} fc={n_fc} ens={n_ens}")

    doc = {
        "generated_at": utc_now_iso(),
        "source": "Open-Meteo Flood API (https://flood-api.open-meteo.com/v1/flood), GloFAS v4, no key",
        "window": "past 30 days + 30-day forecast, daily river discharge (m3/s), UTC",
        "note": ("Discharge is for the GloFAS 0.05-deg river-network grid cell nearest each "
                 "point, NOT an official CWC gauge. Past days are GloFAS consolidated "
                 "reanalysis (class OBSERVED, model reanalysis labelling); future days are "
                 "the GloFAS ensemble forecast (class FORECAST). Ensemble min/median/mean/max "
                 "attached where the API provides them."),
        "degraded_points": degraded,
        "points": points_out,
    }
    save_json(DATA_DIR / "discharge.json", doc)

    # ---- verification ----
    check = __import__("json").load(open(DATA_DIR / "discharge.json", encoding="utf-8"))
    with_data = [pt for pt in check["points"] if len(pt["daily"]) >= 20]
    assert len(with_data) >= 6, f"need >= 6 points with data, got {len(with_data)}"
    for pt in check["points"]:
        for r in pt["daily"]:
            assert {"source", "retrieved_at", "class"} <= set(r), f"{pt['id']}: provenance missing"
    print(f"OK data/discharge.json: {len(with_data)}/{len(check['points'])} points with data, "
          f"degraded={degraded or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
