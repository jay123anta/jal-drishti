"""STEP 1 - fetch hourly precipitation (REAL, LIVE) from Open-Meteo.

Window: past 72 h + next 48 h, hourly, for ~13 named grid points covering
the Nagaland hills, the Meghalaya border south of Guwahati, the Barak
catchment (Silchar + upstream Manipur hills), and the catchments of the
Step-2 river points.

Provenance: hours at or before retrieval time -> class OBSERVED with source
"Open-Meteo forecast product (past hours, model analysis)"; hours after
retrieval -> class FORECAST, source "Open-Meteo forecast product".

Raw responses -> data/raw/rainfall_<id>.json
Normalised    -> data/rainfall.json

Fallback (only on live failure after 3 retries): deterministic SIMULATED
fixture in data/fixtures/, clearly labelled, wired in automatically.
"""

import datetime
import math
import sys

from common import (DATA_DIR, FIXTURES_DIR, FORECAST, OBSERVED, RAW_DIR,
                    SIMULATED, fetch_json, save_failed_request, save_json,
                    utc_now_iso)

API_URL = "https://api.open-meteo.com/v1/forecast"

# Named grid points. Names are real towns used as catchment anchors; the
# rainfall value is for the model grid cell containing these coordinates.
GRID_POINTS = [
    {"id": "wokha",       "name": "Wokha",       "lat": 26.10, "lon": 94.26, "region": "Nagaland hills", "catchment": "Doyang (Dhansiri/Brahmaputra)"},
    {"id": "mokokchung",  "name": "Mokokchung",  "lat": 26.32, "lon": 94.52, "region": "Nagaland hills", "catchment": "Dikhow headwaters"},
    {"id": "zunheboto",   "name": "Zunheboto",   "lat": 26.01, "lon": 94.52, "region": "Nagaland hills", "catchment": "Dikhow/Doyang divide"},
    {"id": "nongpoh",     "name": "Nongpoh",     "lat": 25.90, "lon": 91.88, "region": "Meghalaya border south of Guwahati", "catchment": "Umtrew/Digaru (south-bank Brahmaputra)"},
    {"id": "byrnihat",    "name": "Byrnihat",    "lat": 26.05, "lon": 91.87, "region": "Meghalaya border south of Guwahati", "catchment": "Umtrew (south-bank Brahmaputra)"},
    {"id": "umiam",       "name": "Umiam (Shillong plateau)", "lat": 25.67, "lon": 91.91, "region": "Meghalaya plateau", "catchment": "Umiam/Umtrew"},
    {"id": "silchar",     "name": "Silchar",     "lat": 24.83, "lon": 92.78, "region": "Barak valley", "catchment": "Barak (valley floor)"},
    {"id": "jiribam",     "name": "Jiribam",     "lat": 24.80, "lon": 93.12, "region": "Manipur (Barak upstream)", "catchment": "Barak middle reach"},
    {"id": "tamenglong",  "name": "Tamenglong",  "lat": 24.99, "lon": 93.50, "region": "Manipur hills", "catchment": "Barak headwaters"},
    {"id": "haflong",     "name": "Haflong",     "lat": 25.17, "lon": 93.02, "region": "Dima Hasao hills", "catchment": "Jatinga/Barak north slope"},
    {"id": "hamren",      "name": "Hamren",      "lat": 25.70, "lon": 92.85, "region": "Karbi Anglong hills", "catchment": "Kopili headwaters"},
    {"id": "bhalukpong",  "name": "Bhalukpong",  "lat": 27.01, "lon": 92.64, "region": "Arunachal foothills", "catchment": "Jia Bharali (Kameng)"},
    {"id": "sivasagar",   "name": "Sivasagar",   "lat": 26.98, "lon": 94.63, "region": "Upper Assam plains", "catchment": "Dikhow lower reach"},
]

PARAMS_TEMPLATE = {
    "hourly": "precipitation",
    "past_days": 3,       # covers past 72 h
    "forecast_days": 3,   # covers next 48 h with margin
    "timezone": "UTC",
}


def simulated_hours(point: dict, now: datetime.datetime) -> list[dict]:
    """Deterministic clearly-labelled SIMULATED fallback series."""
    seed = sum(ord(c) for c in point["id"])
    hours = []
    start = now - datetime.timedelta(hours=72)
    for i in range(72 + 48):
        t = start + datetime.timedelta(hours=i)
        # deterministic pseudo-monsoon pattern, NOT real weather
        mm = max(0.0, 4.0 * math.sin((i + seed) / 9.0) + 2.5 * math.sin((i + seed) / 31.0))
        hours.append({
            "time": t.strftime("%Y-%m-%dT%H:%MZ"),
            "precipitation_mm": round(mm, 2),
            "class": SIMULATED,
            "source": "SIMULATED fixture (live Open-Meteo call failed)",
            "retrieved_at": utc_now_iso(),
            "simulated": True,
        })
    return hours


def normalise_hours(api: dict, retrieved_at: str) -> list[dict]:
    now = datetime.datetime.now(datetime.timezone.utc)
    lo = now - datetime.timedelta(hours=72)
    hi = now + datetime.timedelta(hours=48)
    out = []
    times = api["hourly"]["time"]
    precip = api["hourly"]["precipitation"]
    for t_str, mm in zip(times, precip):
        t = datetime.datetime.fromisoformat(t_str).replace(tzinfo=datetime.timezone.utc)
        if t < lo or t > hi or mm is None:
            continue
        past = t <= now
        out.append({
            "time": t.strftime("%Y-%m-%dT%H:%MZ"),
            "precipitation_mm": mm,
            "class": OBSERVED if past else FORECAST,
            "source": ("Open-Meteo forecast product (past hours, model analysis)"
                       if past else "Open-Meteo forecast product"),
            "retrieved_at": retrieved_at,
        })
    return out


def main() -> int:
    points_out = []
    degraded = []
    for p in GRID_POINTS:
        params = dict(PARAMS_TEMPLATE, latitude=p["lat"], longitude=p["lon"])
        retrieved_at = utc_now_iso()
        try:
            api = fetch_json(API_URL, params)
            save_json(RAW_DIR / f"rainfall_{p['id']}.json", api)
            hours = normalise_hours(api, retrieved_at)
            elev = api.get("elevation")
            live = True
        except RuntimeError as err:
            save_failed_request(f"rainfall_{p['id']}", API_URL, params, str(err))
            now = datetime.datetime.now(datetime.timezone.utc)
            hours = simulated_hours(p, now)
            fixture = dict(p, simulated=True, hourly=hours)
            save_json(FIXTURES_DIR / f"rainfall_{p['id']}.json", fixture)
            elev = None
            live = False
            degraded.append(p["id"])
        points_out.append({
            **p,
            "api_elevation_m": elev,
            "live": live,
            "retrieved_at": retrieved_at,
            "hourly": hours,
        })
        n_obs = sum(1 for h in hours if h["class"] == OBSERVED)
        n_fc = sum(1 for h in hours if h["class"] == FORECAST)
        print(f"  {p['id']:12s} live={live} observed_hours={n_obs} forecast_hours={n_fc}")

    doc = {
        "generated_at": utc_now_iso(),
        "source": "Open-Meteo forecast API (https://api.open-meteo.com/v1/forecast), no key, free tier",
        "window": "past 72 h + next 48 h, hourly precipitation, UTC",
        "note": ("Values are for the Open-Meteo model grid cell containing each named point. "
                 "Past hours are the model's analysis of what fell (class OBSERVED); future "
                 "hours are model forecast (class FORECAST). No gauge data is used or claimed."),
        "degraded_points": degraded,
        "points": points_out,
    }
    save_json(DATA_DIR / "rainfall.json", doc)

    # ---- verification (step is done only if this passes) ----
    check = __import__("json").load(open(DATA_DIR / "rainfall.json", encoding="utf-8"))
    assert len(check["points"]) >= 8, "need >= 8 points"
    for pt in check["points"]:
        classes = {h["class"] for h in pt["hourly"]}
        assert classes & {OBSERVED, SIMULATED}, f"{pt['id']}: no past data"
        assert classes & {FORECAST, SIMULATED}, f"{pt['id']}: no future data"
        for h in pt["hourly"]:
            assert {"source", "retrieved_at", "class"} <= set(h), f"{pt['id']}: provenance missing"
    print(f"OK data/rainfall.json: {len(check['points'])} points, degraded={degraded or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
