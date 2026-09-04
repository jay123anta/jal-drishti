"""STEP 4 - classify village flood risk (SIMULATED thresholds, REAL inputs).

Reads:  data/rainfall.json (Step 1), data/discharge.json (Step 2),
        data/villages.json (Step 3)
Writes: public/villages_status.json  (flagship artifact)
        public/rivers_status.json

The heuristic is deliberately simple and UNVALIDATED - see HEURISTIC.md for
the exact rules and known failure modes. Risk classes are class SIMULATED
and carry the label "DEMO - UNVALIDATED CLASSIFICATION". Input signals keep
their real provenance class (OBSERVED / FORECAST) end to end.

UPGRADE v2, STEP A2 - replay mode:
    python classify_risk.py --replay 2026-07-10 2026-08-05
runs the SAME unchanged scoring rules (rain_score / discharge_score /
risk_class, same IDW-of-2-nearest + nearest-river-point geometry) over the
archived window fetched by fetch_archive.py, at 6-hour steps, and writes
public/replay_2026-07.json. Every risk value is class SIMULATED with
replay: true; the rainfall/discharge inputs keep class OBSERVED (they are
reanalysis retrievals of the past). The replay rain-point set additionally
contains Mon town (the cloudburst district) - the RULES are unchanged, the
input grid gains the point closest to where the rain actually fell.
"""

import datetime
import math
import sys

from common import (DATA_DIR, FORECAST, OBSERVED, PUBLIC_DIR, SIMULATED,
                    load_json, save_json, utc_now_iso)

HEURISTIC_SOURCE = ("JalDrishti demo heuristic v0.1 (see HEURISTIC.md; discharge baseline = "
                    "2015-2025 same-season GloFAS distribution where history exists, else "
                    "trailing 30 days) - placeholder, unvalidated")
SEASONAL_SRC = ("derived: percentile of the value within the 2015-2025 GloFAS reanalysis "
                "distribution for the same day-of-year +/- 15 d (heuristic v0.1 seasonal "
                "baseline; NOT a danger level)")
DISCLAIMER = "DEMO - UNVALIDATED CLASSIFICATION - NOT A WARNING SYSTEM"

# --- placeholder cutoffs (mm / percentile), documented in HEURISTIC.md ---
RAIN_YELLOW = {"h24": 30.0, "h48": 60.0}
RAIN_RED = {"h24": 60.0, "h48": 100.0}
DISCH_YELLOW_PCTL = 0.70
DISCH_RED_PCTL = 0.90
N_RAIN_NEIGHBOURS = 2


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


_SEASONAL = {}


def _seasonal_series(river_id: str):
    """(day-of-year array, discharge array) from 2015-2025 history, cached."""
    if river_id in _SEASONAL:
        return _SEASONAL[river_id]
    d = DATA_DIR / "history" / "discharge" / river_id
    files = sorted(p for p in d.glob("*.parquet") if p.stem != "2026") if d.exists() else []
    if not files:
        _SEASONAL[river_id] = None
        return None
    import numpy as np
    import pandas as pd
    df = pd.concat(pd.read_parquet(p) for p in files)
    doy = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy()
    q = df["discharge_m3s"].to_numpy(dtype=float)
    _SEASONAL[river_id] = (doy, q)
    return _SEASONAL[river_id]


def seasonal_percentile(river_id: str, q: float, date_str) -> float | None:
    """Heuristic v0.1: percentile of q within the 2015-2025 distribution for
    the same day-of-year +/- 15 d. None if no history for this river cell."""
    s = _seasonal_series(river_id)
    if s is None or q is None:
        return None
    import numpy as np
    doy = datetime.date.fromisoformat(str(date_str)[:10]).timetuple().tm_yday
    dd = np.abs(s[0] - doy)
    dd = np.minimum(dd, 366 - dd)
    ref = s[1][(dd <= 15) & ~np.isnan(s[1])]
    if len(ref) < 60:
        return None
    return round(float((ref <= q).mean()), 3)


def trailing_rain(point: dict, hours: int) -> tuple[float, str]:
    obs = [h for h in point["hourly"] if h["class"] in (OBSERVED, SIMULATED)]
    take = obs[-hours:]
    total = sum(h["precipitation_mm"] for h in take)
    return round(total, 1), take[0]["retrieved_at"] if take else utc_now_iso()


def forecast_rain(point: dict, hours: int) -> float:
    fc = [h for h in point["hourly"] if h["class"] == FORECAST]
    return round(sum(h["precipitation_mm"] for h in fc[:hours]), 1)


# --- Upgrade v2, Step C: trailing-48h trend cutoffs (arbitrary, documented) ---
TREND_RISING_PCT = 5.0     # >= +5% over 48 h -> rising
TREND_FALLING_PCT = -5.0   # <= -5% over 48 h -> falling


def discharge_stats(rp: dict) -> dict:
    """Latest observed discharge + its percentile in the trailing 30-day range
    + the trailing-48h trend (Upgrade v2, Step C)."""
    obs = [r for r in rp["daily"] if r["class"] in (OBSERVED, SIMULATED)]
    window = obs[-30:]
    latest = window[-1]
    vals = sorted(r["discharge_m3s"] for r in window)
    q = latest["discharge_m3s"]
    rank = sum(1 for v in vals if v <= q)
    pctl = rank / len(vals)
    fc_meds = [r["ensemble"].get("median", r["discharge_m3s"])
               for r in rp["daily"] if r["class"] == FORECAST and "ensemble" in r]
    fc_all = [r["discharge_m3s"] for r in rp["daily"] if r["class"] == FORECAST]
    peak_fc = max(fc_meds or fc_all, default=None)

    # trailing-48h trend: latest vs the value 2 daily steps earlier
    trend, trend_pct, q_2d = None, None, None
    if len(window) >= 3:
        q_2d = window[-3]["discharge_m3s"]
        if q_2d and q_2d > 0:
            trend_pct = round(100.0 * (q - q_2d) / q_2d, 1)
            trend = ("rising" if trend_pct >= TREND_RISING_PCT
                     else "falling" if trend_pct <= TREND_FALLING_PCT
                     else "steady")
    seas = seasonal_percentile(rp["id"], q, latest["date"])
    return {
        "latest": latest, "pctl": round(pctl, 3),
        "pctl_seasonal": seas,
        "pctl_used": seas if seas is not None else round(pctl, 3),
        "baseline": "seasonal_2015_2025" if seas is not None else "trailing_30d",
        "win_min": vals[0], "win_max": vals[-1], "n_days": len(vals),
        "peak_forecast": peak_fc,
        "trend": trend, "trend_pct": trend_pct, "q_2d_ago": q_2d,
    }


def rain_score(h24: float, h48: float) -> int:
    if h24 >= RAIN_RED["h24"] or h48 >= RAIN_RED["h48"]:
        return 2
    if h24 >= RAIN_YELLOW["h24"] or h48 >= RAIN_YELLOW["h48"]:
        return 1
    return 0


def discharge_score(pctl: float) -> int:
    if pctl >= DISCH_RED_PCTL:
        return 2
    if pctl >= DISCH_YELLOW_PCTL:
        return 1
    return 0


def risk_class(total: int) -> str:
    return "RED" if total >= 3 else ("YELLOW" if total == 2 else "GREEN")


def pv(value, cls, source, retrieved_at, **extra):
    """Provenance-wrapped value: the only shape numeric outputs are allowed in."""
    return {"value": value, "class": cls, "source": source,
            "retrieved_at": retrieved_at, **extra}


def main() -> int:
    rainfall = load_json(DATA_DIR / "rainfall.json")
    discharge = load_json(DATA_DIR / "discharge.json")
    villages = load_json(DATA_DIR / "villages.json")
    now = utc_now_iso()

    # Model v0 (Step M6): Dikhow-basin villages take their colour from the
    # shipped, walk-forward-validated model; everyone else keeps the
    # heuristic. Degraded model -> heuristic, marked.
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "model"))
    from basins import BASINS, model_names
    model_fcs = {}
    for b in BASINS:
        try:
            model_fcs[b] = load_json(DATA_DIR / model_names(b)["forecast"])
        except (FileNotFoundError, ValueError):
            model_fcs[b] = {"degraded": True,
                            "error": f"{model_names(b)['forecast']} missing "
                                     f"(run backend/model/predict.py --basin {b})"}
    target_to_basin = {BASINS[b]["target"]: b for b in BASINS}

    rain_pts = rainfall["points"]
    river_pts = discharge["points"]
    river_stats = {rp["id"]: discharge_stats(rp) for rp in river_pts}

    # Track T2: official CWC gauge status per river point (public/cwc_stations.json)
    try:
        cwc_doc = load_json(PUBLIC_DIR / "cwc_stations.json")
        cwc_by = {s["poc_river"]: s for s in cwc_doc.get("stations", [])
                  if s.get("poc_river") and not s.get("degraded")}
    except (FileNotFoundError, ValueError):
        cwc_by = {}

    def official_gauge(river_id: str):
        c = cwc_by.get(river_id)
        if not c or c["observed_level_m"].get("value") is None:
            return None
        lvl = c["observed_level_m"]["value"]
        warn, danger = c["warning_level_m"]["value"], c["danger_level_m"]["value"]
        status = ("above_danger" if danger is not None and lvl >= danger else
                  "above_warning" if warn is not None and lvl >= warn else "below_warning")
        return {
            "station": c["aff_station"], "plain_name": c["plain_name"],
            "observed_level_m": c["observed_level_m"],
            "warning_level_m": c["warning_level_m"], "danger_level_m": c["danger_level_m"],
            "status": pv(status, OBSERVED,
                         "derived: CWC observed level vs the official warning/danger marks "
                         "(CWC Advisory Flood Forecast public dissemination portal)",
                         c["observed_level_m"]["retrieved_at"]),
            "cwc_forecast_peak_m": c["cwc_forecast_peak_m"],
            "cwc_forecast_crosses_warning_at_ist": c.get("cwc_forecast_crosses_warning_at_ist"),
        }

    # ---------- rivers_status.json ----------
    rivers_out = []
    for rp in river_pts:
        st = river_stats[rp["id"]]
        latest = st["latest"]
        rivers_out.append({
            "id": rp["id"], "river": rp["river"], "site": rp["site"],
            "lat": rp["lat"], "lon": rp["lon"],
            "grid_lat": rp.get("grid_lat"), "grid_lon": rp.get("grid_lon"),
            "snap_note": rp.get("snap_note"),
            "live": rp.get("live", False),
            "discharge_latest_m3s": pv(latest["discharge_m3s"], latest["class"],
                                       latest["source"], latest["retrieved_at"],
                                       date=latest["date"], unit="m3/s"),
            "discharge_percentile_30d": pv(st["pctl"], latest["class"],
                                           "derived: percentile of latest value within its own "
                                           "trailing 30-day GloFAS series (NOT a danger level)",
                                           latest["retrieved_at"],
                                           window_days=st["n_days"],
                                           window_min_m3s=st["win_min"],
                                           window_max_m3s=st["win_max"]),
            "discharge_peak_forecast_m3s": pv(st["peak_forecast"], FORECAST,
                                              "GloFAS ensemble median, max over forecast horizon, "
                                              "via Open-Meteo Flood API", latest["retrieved_at"],
                                              unit="m3/s"),
            "discharge_percentile_seasonal": pv(st["pctl_seasonal"], latest["class"],
                                                SEASONAL_SRC, latest["retrieved_at"],
                                                baseline_used=st["baseline"]),
            "trend_48h": pv(st["trend"], latest["class"],
                            "derived: latest GloFAS reanalysis value vs 2 days earlier "
                            f"(rising >= +{TREND_RISING_PCT:.0f}%, falling <= "
                            f"{TREND_FALLING_PCT:.0f}%, else steady; arbitrary demo cutoffs)",
                            latest["retrieved_at"],
                            pct_change_48h=st["trend_pct"],
                            from_m3s=st["q_2d_ago"],
                            to_m3s=latest["discharge_m3s"]),
            "official_gauge": official_gauge(rp["id"]),
            "daily": rp["daily"],  # full per-day records keep their own provenance
        })

    save_json(PUBLIC_DIR / "rivers_status.json", {
        "generated_at": now,
        "disclaimer": DISCLAIMER,
        "note": discharge["note"],
        "rivers": rivers_out,
    })

    # ---------- villages_status.json ----------
    villages_out = []
    for v in villages["villages"]:
        # nearest N rainfall grid points, inverse-distance weighted
        ranked = sorted(rain_pts, key=lambda p: haversine_km(v["lat"], v["lon"], p["lat"], p["lon"]))
        near = ranked[:N_RAIN_NEIGHBOURS]
        dists = [max(haversine_km(v["lat"], v["lon"], p["lat"], p["lon"]), 1.0) for p in near]
        wts = [1.0 / d for d in dists]
        wsum = sum(wts)
        wts = [w / wsum for w in wts]

        def idw(fn):
            return round(sum(w * fn(p) for w, p in zip(wts, near)), 1)

        h24 = idw(lambda p: trailing_rain(p, 24)[0])
        h48 = idw(lambda p: trailing_rain(p, 48)[0])
        next24 = idw(lambda p: forecast_rain(p, 24))
        rain_cls = OBSERVED if all(p.get("live") for p in near) else SIMULATED
        rain_src = ("IDW mean of " + " + ".join(p["id"] for p in near) +
                    ", Open-Meteo forecast product (past hours, model analysis)")
        rain_ra = near[0]["retrieved_at"]

        # nearest river point
        rp = min(river_pts, key=lambda p: haversine_km(v["lat"], v["lon"], p["lat"], p["lon"]))
        rp_km = haversine_km(v["lat"], v["lon"], rp["lat"], rp["lon"])
        st = river_stats[rp["id"]]
        latest = st["latest"]

        a = rain_score(h24, h48)
        b = discharge_score(st["pctl_used"])
        cls = risk_class(a + b)

        # method selection: a village belongs to a basin if its nearest river
        # point is that basin's target; the basin's shipped model (if live)
        # gives the colour, otherwise the heuristic (marked degraded)
        basin = target_to_basin.get(rp["id"])
        model_fc = model_fcs.get(basin) if basin else None
        model_ok = bool(model_fc) and not model_fc.get("degraded", True)
        if basin and model_ok:
            risk_obj = pv(model_fc["colour"]["value"], SIMULATED,
                          model_fc["colour"]["source"], now,
                          label=DISCLAIMER, method=model_fc["model"],
                          forecast_horizon=model_fc["forecast_horizon"],
                          basis=model_fc["basis_plain"],
                          p_exceed_h1=model_fc["colour"]["p_exceed_h1"],
                          model_card=model_fc["model_card"],
                          target_caveat=model_fc["target_caveat"])
        else:
            extra = {}
            if basin and not model_ok:
                extra = {"model_degraded": model_fc.get("error", "unknown")}
            risk_obj = pv(cls, SIMULATED, HEURISTIC_SOURCE, now,
                          label=DISCLAIMER, method="heuristic", **extra)

        coord_observed = v["coordinate_precision"] == "osm_node"
        villages_out.append({
            "name": v["name"], "district": v["district"],
            "location": {
                "lat": v["lat"], "lon": v["lon"],
                "class": OBSERVED if coord_observed else SIMULATED,
                "source": v["source"],
                "coordinate_precision": v["coordinate_precision"],
                "retrieved_at": v.get("retrieved_at", now),
            },
            "risk": risk_obj,
            "scores": {
                "rain_score": pv(a, SIMULATED, HEURISTIC_SOURCE, now,
                                 note="0/1/2 from placeholder mm cutoffs"),
                "discharge_score": pv(b, SIMULATED, HEURISTIC_SOURCE, now,
                                      note="0/1/2 from placeholder percentile cutoffs",
                                      baseline_used=st["baseline"]),
                "combined": pv(a + b, SIMULATED, HEURISTIC_SOURCE, now,
                               note="sum; 0-1 GREEN, 2 YELLOW, 3-4 RED"),
            },
            "signals": {
                "rain_trailing_24h_mm": pv(h24, rain_cls, rain_src, rain_ra, unit="mm",
                                           derivation="trailing 24 h cumulative"),
                "rain_trailing_48h_mm": pv(h48, rain_cls, rain_src, rain_ra, unit="mm",
                                           derivation="trailing 48 h cumulative"),
                "rain_forecast_next24h_mm": pv(next24, FORECAST,
                                               "Open-Meteo forecast product (IDW of " +
                                               " + ".join(p["id"] for p in near) + ")",
                                               rain_ra, unit="mm",
                                               derivation="next 24 h cumulative forecast"),
                "discharge_latest_m3s": pv(latest["discharge_m3s"], latest["class"],
                                           latest["source"], latest["retrieved_at"],
                                           unit="m3/s", date=latest["date"],
                                           river_point=rp["id"]),
                "discharge_percentile_30d": pv(st["pctl"], latest["class"],
                                               "derived: percentile within own trailing 30-day "
                                               "GloFAS series (NOT a danger level)",
                                               latest["retrieved_at"],
                                               river_point=rp["id"]),
                "discharge_percentile_seasonal": pv(st["pctl_seasonal"], latest["class"],
                                                    SEASONAL_SRC, latest["retrieved_at"],
                                                    river_point=rp["id"],
                                                    baseline_used=st["baseline"]),
            },
            "inputs_used": {
                "rain_points": [{"id": p["id"], "name": p["name"],
                                 "distance_km": round(haversine_km(v["lat"], v["lon"], p["lat"], p["lon"]), 1),
                                 "weight": round(w, 3)}
                                for p, w in zip(near, wts)],
                "river_point": {"id": rp["id"], "river": rp["river"], "site": rp["site"],
                                "distance_km": round(rp_km, 1)},
            },
        })

    n_model = sum(1 for x in villages_out if x["risk"]["method"].startswith("model-v0-"))
    save_json(PUBLIC_DIR / "villages_status.json", {
        "generated_at": now,
        "disclaimer": DISCLAIMER,
        "heuristic": HEURISTIC_SOURCE,
        "model_status": {b: {
            "model": model_names(b)["method"] + (
                f" (shipped: {model_fcs[b].get('shipped')}, walk-forward validated)"
                if not model_fcs[b].get("degraded") else ""),
            "basin_label": BASINS[b]["label"],
            "degraded": bool(model_fcs[b].get("degraded")),
            "error": model_fcs[b].get("error"),
            "villages_covered": sum(1 for x in villages_out
                                    if x["risk"]["method"] == model_names(b)["method"]),
            "target_caveat": model_fcs[b].get("target_caveat",
                                              "GloFAS reanalysis target - modelled "
                                              "product, not observed river data"),
            "model_card": model_names(b)["card"],
            "test_file": model_names(b)["test"], "drift_file": model_names(b)["drift"],
        } for b in BASINS},
        "counts": {c: sum(1 for x in villages_out if x["risk"]["value"] == c)
                   for c in ("GREEN", "YELLOW", "RED")},
        "villages_note": villages["note"],
        "villages": villages_out,
    })

    counts = {c: sum(1 for x in villages_out if x["risk"]["value"] == c)
              for c in ("GREEN", "YELLOW", "RED")}
    print(f"OK public/villages_status.json: {len(villages_out)} villages -> {counts} "
          f"({n_model} via basin models "
          f"{ {b: sum(1 for x in villages_out if x['risk']['method'] == model_names(b)['method']) for b in BASINS} }, "
          f"{len(villages_out) - n_model} via heuristic)")
    print(f"OK public/rivers_status.json: {len(rivers_out)} river points")
    return 0


# ---------------------------------------------------------------------------
# UPGRADE v2, STEP A2 - retrospective replay over the archived window.
# The scoring functions above are used AS-IS; nothing below alters them.
# ---------------------------------------------------------------------------

ARCHIVE_TAG = "2026-07"
ARCHIVE_DIR = DATA_DIR / "archive" / ARCHIVE_TAG
REPLAY_STEP_HOURS = 6
REPLAY_SOURCE = HEURISTIC_SOURCE + " - retrospective replay over archived inputs"


def parse_utc(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s.replace("Z", "")).replace(
        tzinfo=datetime.timezone.utc)


def replay_rain_series(pt: dict) -> tuple[list[datetime.datetime], list[float]]:
    """Hour timestamps + cumulative precip (prefix sums) for O(1) trailing sums."""
    times, cum, total = [], [], 0.0
    for h in pt["hourly"]:
        times.append(parse_utc(h["time"]))
        total += h["precipitation_mm"]
        cum.append(total)
    return times, cum


def trailing_sum(times, cum, t: datetime.datetime, hours: int) -> float:
    """Sum of hourly precip with timestamp in (t - hours, t] - mirrors the
    live trailing_rain() semantics (last N hourly entries up to now)."""
    import bisect
    hi = bisect.bisect_right(times, t)
    lo = bisect.bisect_right(times, t - datetime.timedelta(hours=hours))
    if hi == 0:
        return 0.0
    return round(cum[hi - 1] - (cum[lo - 1] if lo > 0 else 0.0), 1)


def replay_discharge_pctl(daily: list[dict], date: datetime.date) -> tuple[float, float] | None:
    """(discharge, percentile) for `date` within its trailing 30-day window,
    using the same rank/len formula as discharge_stats()."""
    by_date = {datetime.date.fromisoformat(r["date"]): r["discharge_m3s"] for r in daily}
    if date not in by_date:
        prior = [d for d in by_date if d <= date]
        if not prior:
            return None
        date = max(prior)
    window = [by_date[d] for d in by_date
              if date - datetime.timedelta(days=29) <= d <= date]
    if not window:
        return None
    q = by_date[date]
    rank = sum(1 for v in sorted(window) if v <= q)
    return q, round(rank / len(window), 3)


def replay_main(start_str: str, end_str: str) -> int:
    rain_doc = load_json(ARCHIVE_DIR / "archive_rainfall.json")
    disch_doc = load_json(ARCHIVE_DIR / "archive_discharge.json")
    villages = load_json(DATA_DIR / "villages.json")
    now = utc_now_iso()

    rain_pts = rain_doc["points"]
    river_pts = disch_doc["points"]
    rain_series = {p["id"]: replay_rain_series(p) for p in rain_pts}

    start = parse_utc(start_str + "T00:00")
    end = parse_utc(end_str + "T18:00")
    steps = []
    t = start
    while t <= end:
        steps.append(t)
        t += datetime.timedelta(hours=REPLAY_STEP_HOURS)
    times_iso = [t.strftime("%Y-%m-%dT%H:%MZ") for t in steps]

    villages_out = []
    step_counts = [{"GREEN": 0, "YELLOW": 0, "RED": 0} for _ in steps]
    replay_seasonal_cache = {}
    for v in villages["villages"]:
        ranked = sorted(rain_pts, key=lambda p: haversine_km(v["lat"], v["lon"], p["lat"], p["lon"]))
        near = ranked[:N_RAIN_NEIGHBOURS]
        dists = [max(haversine_km(v["lat"], v["lon"], p["lat"], p["lon"]), 1.0) for p in near]
        wts = [1.0 / d for d in dists]
        wsum = sum(wts)
        wts = [w / wsum for w in wts]
        rain_src = ("IDW mean of " + " + ".join(p["id"] for p in near) +
                    ", " + rain_doc["source"])
        rain_ra = near[0]["retrieved_at"]

        rp = min(river_pts, key=lambda p: haversine_km(v["lat"], v["lon"], p["lat"], p["lon"]))
        disch_ra = rp["retrieved_at"]

        entries = []
        for i, t in enumerate(steps):
            h24 = round(sum(w * trailing_sum(*rain_series[p["id"]], t, 24)
                            for w, p in zip(wts, near)), 1)
            h48 = round(sum(w * trailing_sum(*rain_series[p["id"]], t, 48)
                            for w, p in zip(wts, near)), 1)
            dq = replay_discharge_pctl(rp["daily"], t.date())
            if dq is None:
                continue
            q, pctl = dq
            seas = replay_seasonal_cache.get((rp["id"], t.date()))
            if seas is None and (rp["id"], t.date()) not in replay_seasonal_cache:
                seas = seasonal_percentile(rp["id"], q, t.date())
                replay_seasonal_cache[(rp["id"], t.date())] = seas
            pctl_used = seas if seas is not None else pctl
            a = rain_score(h24, h48)
            b = discharge_score(pctl_used)
            cls = risk_class(a + b)
            step_counts[i][cls] += 1
            entries.append({
                "t": times_iso[i],
                "risk": {"value": cls, "class": SIMULATED, "source": REPLAY_SOURCE,
                         "retrieved_at": now, "replay": True, "label": DISCLAIMER},
                "rain24": pv(h24, OBSERVED, rain_src, rain_ra, unit="mm", replay=True),
                "rain48": pv(h48, OBSERVED, rain_src, rain_ra, unit="mm", replay=True),
                "discharge_m3s": pv(q, OBSERVED, rp["daily"][0]["source"], disch_ra,
                                    unit="m3/s", river_point=rp["id"], replay=True),
                "discharge_pctl": pv(pctl_used, OBSERVED,
                                     SEASONAL_SRC if seas is not None else
                                     "derived: percentile within trailing 30-day GloFAS "
                                     "reanalysis window ending that day (NOT a danger level)",
                                     disch_ra, river_point=rp["id"], replay=True,
                                     baseline_used="seasonal_2015_2025" if seas is not None
                                     else "trailing_30d"),
            })
        villages_out.append({
            "name": v["name"], "district": v["district"],
            "lat": v["lat"], "lon": v["lon"],
            "rain_points": [p["id"] for p in near],
            "river_point": rp["id"],
            "steps": entries,
        })

    doc = {
        "generated_at": now,
        "replay": True,
        "disclaimer": DISCLAIMER,
        "heuristic": REPLAY_SOURCE,
        "event_note": ("Retrospective replay around the 19 July 2026 cloudburst in Mon "
                       "district, Nagaland, and the flooding that followed in Sivasagar, "
                       "Charaideo, Jorhat and Golaghat districts via the south-bank "
                       "tributaries. A real disaster with loss of life. This replay shows "
                       "what the demo rules would have computed from archived model data; "
                       "it is NOT evidence that correct warnings would have been issued."),
        "window": {"start": start_str, "end": end_str, "step_hours": REPLAY_STEP_HOURS},
        "times": times_iso,
        "step_counts": step_counts,
        "rain_points_used": [{"id": p["id"], "name": p["name"], "lat": p["lat"],
                              "lon": p["lon"]} for p in rain_pts],
        "villages": villages_out,
    }
    out_path = PUBLIC_DIR / f"replay_{ARCHIVE_TAG}.json"
    # compact dump: ~5,400 village-steps with full per-value provenance
    import json as _json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        _json.dump(doc, fh, separators=(",", ":"), ensure_ascii=False)

    # ---- verification ----
    check = load_json(out_path)
    assert check["replay"] is True and "UNVALIDATED" in check["disclaimer"]
    assert len(check["times"]) == len(steps) and len(check["villages"]) == len(villages["villages"])
    n_entries = 0
    for vv in check["villages"]:
        assert len(vv["steps"]) == len(steps), f"{vv['name']}: incomplete series"
        for e in vv["steps"]:
            r = e["risk"]
            assert r["value"] in {"GREEN", "YELLOW", "RED"} and r["class"] == SIMULATED
            assert r["replay"] is True and "UNVALIDATED" in r["label"]
            for k in ("rain24", "rain48", "discharge_m3s", "discharge_pctl"):
                assert {"value", "source", "retrieved_at", "class"} <= set(e[k])
                assert e[k]["class"] == OBSERVED
            n_entries += 1
    peak = max(range(len(steps)), key=lambda i: step_counts[i]["RED"] * 1000 + step_counts[i]["YELLOW"])
    print(f"OK public/replay_{ARCHIVE_TAG}.json: {len(check['villages'])} villages x "
          f"{len(steps)} steps ({n_entries} entries), peak step {times_iso[peak]} -> "
          f"{step_counts[peak]}")
    return 0


if __name__ == "__main__":
    if "--replay" in sys.argv:
        i = sys.argv.index("--replay")
        sys.exit(replay_main(sys.argv[i + 1], sys.argv[i + 2]))
    sys.exit(main())
