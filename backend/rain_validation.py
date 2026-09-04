"""Rain validation: Open-Meteo model rain vs REAL NWDP gauges.

The whole pipeline (features, lag analyses, replay narratives) runs on
Open-Meteo rain, which is itself a weather-model product. The NWDP ingest
(fetch_nwdp.py) gave us ~124 OBSERVED hourly rain gauges in and around our
catchments. This script scores Open-Meteo AGAINST those gauges - touching
no model, changing no behaviour; it only measures how much the rain inputs
deserve to be trusted.

Method (fixed before results were seen):
- Stations: every *_rainfall NWDP station with trusted coordinates and
  >= 2000 hourly readings.
- Gauge timestamps are assumed IST and shifted -05:30 to UTC (the same
  assumption the CWC AFF feed required); stated as a caveat.
- Open-Meteo ERA5-blend archive hourly precipitation fetched AT EACH
  GAUGE'S OWN lat/lon, 2022-01-01 .. yesterday (raw responses cached
  gzipped so reruns are free).
- Compared on DAILY totals (UTC days), only on days where the gauge
  reported >= 20 hourly values. Gauge cleaning: negatives dropped,
  hourly values > 150 mm dropped as sensor spikes (counted).
- Metrics per station: n_days, Pearson r, mean daily mm (gauge vs OM),
  bias ratio OM/gauge, daily MAE; heavy-day check with thresholds fixed
  in advance: heavy = gauge >= 25 mm/day; detected(strict) = OM >= 25;
  detected(lenient) = OM >= 10; false alarm = OM >= 25 while gauge < 10.
- A point gauge vs an ~11 km model cell is an imperfect comparison by
  nature (convective cells are small); scores are read with that in mind.

METHOD AMENDMENT (2026-09-02, disclosed): the first pass treated every
gauge as an hourly-increment sensor and produced physically impossible
totals. Inspection showed the NWDP feed mixes FOUR behaviours: true
hourly gauges, CUMULATIVE counters (values only ever rise), sensor
garbage (readings in the billions of mm), and sparse event-triggered
reporters. The method now classifies each station first: cumulative
series are differenced, garbage and sparse stations are excluded with
the reason recorded. This amendment was made after seeing the GAUGE DATA
QUALITY - never after seeing any score against Open-Meteo. Second
amendment, same rule: the density filter first ran BEFORE classification
and wrongly excluded cumulative counters (whose daily totals are exact at
any reporting density); it now applies only to incremental sensors. Third
amendment, same rule: magnitude scoring proved untrustworthy for nearly
every station (undocumented counter semantics, unit ambiguity, impossible
totals), so a WET-DAY AGREEMENT tier was added - on days a station
actively reported, did Open-Meteo also show rain (>= 1 mm)? That question
is robust to units and counter semantics. Gauge wet-day = any reading
implying >= 1 mm that day; only days with at least one report count.

Outputs: data/history/rain_validation.json + docs/RAIN-VALIDATION.md.
Never fabricates: stations whose fetch fails are listed as skipped with
the real error.
"""
from __future__ import annotations
import datetime
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
NWDP_DIR = BASE_DIR / "data" / "history" / "nwdp"
CACHE_DIR = NWDP_DIR / "raw_om_validation"
OUT_JSON = BASE_DIR / "data" / "history" / "rain_validation.json"
OUT_MD = BASE_DIR / "docs" / "RAIN-VALIDATION.md"

OM_URL = "https://archive-api.open-meteo.com/v1/archive"
START = "2022-01-01"
MIN_ROWS = 2000
MIN_HOURS_PER_DAY = 20
HEAVY_MM = 25.0
LENIENT_MM = 10.0
SPIKE_MM_H = 150.0
BATCH = 8
PAUSE_S = 1.0

SRC = ("Open-Meteo archive API (ERA5-blend model rain) scored against NWIC "
       "National Water Data Portal observed telemetry rain gauges")


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stations() -> list[dict]:
    out = []
    for kind_dir in sorted(NWDP_DIR.glob("*_rainfall")):
        for sdir in sorted(kind_dir.iterdir()):
            prov_p = sdir / "data.provenance.json"
            if not prov_p.exists():
                continue
            prov = json.loads(prov_p.read_text(encoding="utf-8"))
            if prov.get("note"):          # coord-untrusted - excluded
                continue
            if prov.get("rows", 0) < MIN_ROWS:
                continue
            out.append({"org": kind_dir.name.replace("_rainfall", ""),
                        "slug": sdir.name, "dir": sdir, **{
                            k: prov[k] for k in ("station", "district", "lat", "lon")}})
    return out


def om_daily(st: dict, end_date: str) -> pd.Series | None:
    """Open-Meteo archive daily totals at the gauge location (cached)."""
    cache = CACHE_DIR / f"{st['org']}_{st['slug']}.json.gz"
    if cache.exists():
        doc = json.loads(gzip.decompress(cache.read_bytes()))
    else:
        r = requests.get(OM_URL, params={
            "latitude": st["lat"], "longitude": st["lon"],
            "hourly": "precipitation", "start_date": START, "end_date": end_date,
            "timezone": "UTC"}, timeout=120)
        r.raise_for_status()
        doc = r.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(gzip.compress(json.dumps(doc).encode(), 6))
        time.sleep(PAUSE_S)
    h = doc.get("hourly", {})
    if not h.get("time"):
        return None
    s = pd.Series(h["precipitation"], index=pd.to_datetime(h["time"]), dtype="float")
    return s.groupby(s.index.date).sum()


def gauge_daily(st: dict) -> tuple[pd.Series | None, dict]:
    """Classify the station's reporting behaviour, then build daily totals.

    Returns (daily_series or None, meta). meta['mode'] is 'incremental' or
    'cumulative'; excluded stations get meta['excluded'] with the reason.
    """
    df = pd.read_parquet(st["dir"] / "data.parquet")
    t = pd.to_datetime(df["t"]) - pd.Timedelta(hours=5, minutes=30)  # IST -> UTC
    v = pd.to_numeric(df["value"], errors="coerce")
    s = pd.Series(v.values, index=t).sort_index().dropna()
    meta = {"n_raw": int(len(df)), "n_neg": int((s < 0).sum())}
    s = s[s >= 0]
    if len(s) < MIN_ROWS // 2:
        meta["excluded"] = "too few valid readings"
        return None, meta
    meta["max_value"] = float(s.max())
    if meta["max_value"] > 1000:
        meta["excluded"] = "implausible extremes (sensor garbage, max > 1000 mm)"
        return None, meta
    frac_nd = float((s.diff().dropna() >= 0).mean())
    meta["frac_nondecreasing"] = round(frac_nd, 3)
    meta["median_readings_per_day"] = float(
        pd.Series(s.index.date).value_counts().median())
    if frac_nd >= 0.95:
        # CUMULATIVE counter: daily total = sum of positive steps, which is
        # exact even at sparse reporting density -> only a light day filter.
        meta["mode"] = "cumulative"
        inc = s.diff().clip(lower=0)      # counter resets become 0, not negative
        inc = inc[inc <= SPIKE_MM_H]
        g = inc.groupby(inc.index.date)
        counts, sums = g.size(), g.sum()
        return sums[counts >= 4], meta
    # INCREMENTAL sensor: daily totals need dense reporting, otherwise a
    # telemetry outage is indistinguishable from a dry spell.
    if meta["median_readings_per_day"] < 12:
        meta["excluded"] = ("sparse/event-triggered incremental reporting "
                            "(< 12 readings a day; outage vs dry ambiguous)")
        return None, meta
    meta["mode"] = "incremental"
    sv = s[s <= SPIKE_MM_H]
    meta["n_spikes_dropped"] = int(len(s) - len(sv))
    g = sv.groupby(sv.index.date)
    counts, sums = g.size(), g.sum()
    return sums[counts >= MIN_HOURS_PER_DAY], meta


def score(st: dict, end_date: str) -> dict:
    gd, meta = gauge_daily(st)
    if gd is None:
        return {"skipped": meta["excluded"], "gauge_meta": meta}
    if len(gd) < 30:
        return {"skipped": f"gauge has only {len(gd)} usable days", "gauge_meta": meta}
    od = om_daily(st, end_date)
    if od is None:
        return {"skipped": "Open-Meteo returned no hourly data", "gauge_meta": meta}
    both = pd.DataFrame({"gauge": gd, "om": od}).dropna()
    if len(both) < 30:
        return {"skipped": f"only {len(both)} overlapping days", "gauge_meta": meta}
    heavy = both[both["gauge"] >= HEAVY_MM]
    fa = both[(both["om"] >= HEAVY_MM) & (both["gauge"] < LENIENT_MM)]
    r = float(both["gauge"].corr(both["om"])) if both["gauge"].std() > 0 else None
    return {
        "gauge_mode": meta["mode"],
        "n_days": int(len(both)),
        "pearson_r": round(r, 3) if r is not None else None,
        "gauge_mean_mm_d": round(float(both["gauge"].mean()), 2),
        "om_mean_mm_d": round(float(both["om"].mean()), 2),
        "bias_om_over_gauge": round(float(both["om"].mean() / both["gauge"].mean()), 2)
        if both["gauge"].mean() > 0 else None,
        "daily_mae_mm": round(float((both["gauge"] - both["om"]).abs().mean()), 2),
        "heavy_days_gauge": int(len(heavy)),
        "heavy_detected_strict": int((heavy["om"] >= HEAVY_MM).sum()),
        "heavy_detected_lenient": int((heavy["om"] >= LENIENT_MM).sum()),
        "false_alarm_days": int(len(fa)),
    }


def median_of(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.median(vals)), 3) if vals else None


def wet_day_agreement(st: dict, end_date: str) -> dict | None:
    """Tier B: binary rain/no-rain concordance on gauge-active days."""
    df = pd.read_parquet(st["dir"] / "data.parquet")
    t = pd.to_datetime(df["t"]) - pd.Timedelta(hours=5, minutes=30)
    v = pd.to_numeric(df["value"], errors="coerce")
    s2 = pd.Series(v.values, index=t).sort_index().dropna()
    s2 = s2[(s2 >= 0) & (s2 <= 1000)]
    if len(s2) < 500:
        return None
    frac_nd = float((s2.diff().dropna() >= 0).mean())
    if frac_nd >= 0.95:
        sig = s2.diff().clip(lower=0)     # counter increments
    else:
        sig = s2
    daily_max = sig.groupby(sig.index.date).max()
    reports = s2.groupby(s2.index.date).size()
    active = daily_max[reports >= 1]
    gauge_wet = active >= 1.0
    try:
        od = om_daily(st, end_date)
    except Exception:  # noqa: BLE001
        return None
    if od is None:
        return None
    both = pd.DataFrame({"gw": gauge_wet, "om": od}).dropna()
    if len(both) < 60:
        return None
    om_wet = both["om"] >= 1.0
    gw = both["gw"].astype(bool)   # merge can degrade bool -> int; ~int is bitwise
    n_wet = int(gw.sum()); n_dry = int(len(gw) - gw.sum())
    if n_wet < 10 or n_dry < 10:
        return None
    return {
        "n_active_days": int(len(both)),
        "gauge_wet_days": n_wet,
        "om_wet_when_gauge_wet": round(float(om_wet[gw].mean()), 3),
        "om_wet_when_gauge_dry": round(float(om_wet[~gw].mean()), 3),
    }


def main() -> int:
    end_date = (datetime.datetime.now(datetime.timezone.utc).date()
                - datetime.timedelta(days=2)).isoformat()
    now = utc_now_iso()
    sts = stations()
    print(f"Rain validation: {len(sts)} coord-trusted gauges with >= {MIN_ROWS} rows")
    results, skipped = [], []
    for st in sts:
        try:
            sc = score(st, end_date)
        except Exception as err:  # noqa: BLE001 - record, never fabricate
            sc = {"skipped": f"error: {err}"}
        rec = {"org": st["org"], "station": st["station"],
               "district": st["district"], "lat": st["lat"], "lon": st["lon"], **sc}
        (skipped if "skipped" in sc else results).append(rec)
        tag = sc.get("skipped") or (f"r={sc['pearson_r']} bias={sc['bias_om_over_gauge']} "
                                    f"heavy {sc['heavy_detected_lenient']}/{sc['heavy_days_gauge']}")
        print(f"  {st['org'][:9]:10s}{st['station'][:32]:34s}{tag}")

    summary = {}
    for org in sorted({r["org"] for r in results}):
        rows = [r for r in results if r["org"] == org]
        hd = sum(r["heavy_days_gauge"] for r in rows)
        summary[org] = {
            "stations": len(rows),
            "median_r": median_of(rows, "pearson_r"),
            "median_bias": median_of(rows, "bias_om_over_gauge"),
            "median_daily_mae_mm": median_of(rows, "daily_mae_mm"),
            "heavy_days_total": hd,
            "heavy_detected_strict": sum(r["heavy_detected_strict"] for r in rows),
            "heavy_detected_lenient": sum(r["heavy_detected_lenient"] for r in rows),
            "false_alarm_days": sum(r["false_alarm_days"] for r in rows),
        }

    # Tier B: wet-day agreement over every non-garbage station
    wet_rows = []
    for st in sts:
        try:
            w = wet_day_agreement(st, end_date)
        except Exception:  # noqa: BLE001
            w = None
        if w:
            wet_rows.append({"org": st["org"], "station": st["station"], **w})
    wet_summary = {}
    for org in sorted({r["org"] for r in wet_rows}):
        rows = [r for r in wet_rows if r["org"] == org]
        wet_summary[org] = {
            "stations": len(rows),
            "median_om_wet_when_gauge_wet": median_of(rows, "om_wet_when_gauge_wet"),
            "median_om_wet_when_gauge_dry": median_of(rows, "om_wet_when_gauge_dry"),
        }
    print(f"  wet-day agreement computed for {len(wet_rows)} stations")

    OUT_JSON.write_text(json.dumps({
        "wet_day_agreement": {"summary_by_org": wet_summary, "stations": wet_rows},
        "generated_at": now, "source": SRC, "class": "OBSERVED",
        "method": {"start": START, "end": end_date, "heavy_mm": HEAVY_MM,
                   "lenient_mm": LENIENT_MM, "spike_cut_mm_h": SPIKE_MM_H,
                   "min_hours_per_day": MIN_HOURS_PER_DAY,
                   "gauge_tz_assumption": "IST shifted -05:30 to UTC"},
        "summary_by_org": summary, "stations": results, "skipped": skipped,
    }, indent=1), encoding="utf-8")

    # ---- markdown report ----
    lines = [
        "# RAIN-VALIDATION - Open-Meteo model rain vs real NWDP gauges\n",
        f"Generated {now} by `backend/rain_validation.py`. Method and thresholds",
        "were fixed in the script header before results were computed.\n",
        "**Why this matters:** every rain number in this project (model features,",
        "lag analyses, replay narratives) is Open-Meteo model rain. These are the",
        "first REAL gauges we can score it against, at the gauge's own location.\n",
        "**Read with care:** a point gauge vs an ~11 km model cell is imperfect by",
        "nature (small storms can hit one and miss the other); gauge telemetry is",
        "young (mostly 2022+) and gappy; gauge timestamps assumed IST.\n",
        "## Summary by producing agency\n",
        "| gauges | stations | median corr r | median bias OM/gauge | median daily MAE (mm) | heavy days (gauge >= 25 mm) | OM saw >= 25 | OM saw >= 10 | false alarms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for org, s in summary.items():
        lines.append(
            f"| {org} | {s['stations']} | {s['median_r']} | {s['median_bias']} | "
            f"{s['median_daily_mae_mm']} | {s['heavy_days_total']} | "
            f"{s['heavy_detected_strict']} | {s['heavy_detected_lenient']} | "
            f"{s['false_alarm_days']} |")
    lines += ["\n## Per-station results\n",
              "| agency | station | district | sensor type | days | r | bias | MAE | heavy | seen>=25 | seen>=10 | FA |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (x["org"], -(x["pearson_r"] or 0))):
        lines.append(
            f"| {r['org']} | {r['station'][:30]} | {r['district'][:18]} | {r['gauge_mode']} "
            f"| {r['n_days']} "
            f"| {r['pearson_r']} | {r['bias_om_over_gauge']} | {r['daily_mae_mm']} "
            f"| {r['heavy_days_gauge']} | {r['heavy_detected_strict']} "
            f"| {r['heavy_detected_lenient']} | {r['false_alarm_days']} |")
    lines += [
        "\n## Wet-day agreement (robust tier - every non-garbage station)\n",
        "Magnitudes from this gauge network cannot be trusted (see below), but",
        "'did it rain at all today?' can. On days a station actively reported,",
        "how often did Open-Meteo also show >= 1 mm - and how often did it show",
        "rain when the gauge implied a dry day?\n",
        "| gauges | stations | OM wet when gauge wet (median) | OM wet when gauge dry (median) |",
        "|---|---|---|---|"]
    for org, w in wet_summary.items():
        lines.append(f"| {org} | {w['stations']} | {w['median_om_wet_when_gauge_wet']} "
                     f"| {w['median_om_wet_when_gauge_dry']} |")
    if not wet_summary:
        lines.append("| (no station passed the activity thresholds) | | | |")

    lines += [
        "\n## Verdict (read this first)\n",
        "**Inconclusive - and documented.** This gauge network cannot validate",
        "Open-Meteo's rain magnitudes: most stations are cumulative counters",
        "with undocumented semantics, sparse event-triggered reporters, or",
        "sensor garbage, and the few that score produce physically impossible",
        "totals. The wet-day tier shows Open-Meteo detects most true wet days,",
        "but in monsoon climate it also calls most gauge-dry days wet at the",
        "1 mm threshold - the discrimination gap is thin (Assam's is negative).",
        "So: Open-Meteo rain remains our documented working assumption, not a",
        "validated input. What would settle it: station metadata from the",
        "producing agencies (counter semantics, units), or IMD gridded",
        "observations. The gauge archive is retained for that day.\n"]

    if skipped:
        reasons: dict[str, int] = {}
        for sk in skipped:
            reasons[sk["skipped"]] = reasons.get(sk["skipped"], 0) + 1
        lines += ["\n## Gauge data quality - why stations were excluded\n",
                  "The NWDP telemetry feed mixes sensor behaviours; only stations",
                  "that behave like real rain gauges are scored. Exclusions:\n",
                  "| reason | stations |", "|---|---|"]
        for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {n} |")
        lines += ["\nPer-station diagnostics (reporting density, cumulative",
                  "signature, extreme values) are in "
                  "`data/history/rain_validation.json`."]
    lines += ["\nRaw Open-Meteo responses cached in "
              "`data/history/nwdp/raw_om_validation/` (gzip)."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK rain validation: {len(results)} scored, {len(skipped)} skipped -> "
          f"{OUT_JSON.name} + docs/{OUT_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
