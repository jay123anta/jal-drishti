"""GloFAS vs OBSERVED - how far the modelled river sits from the real one.

Every model in this project is trained and scored against GloFAS v4 discharge,
which is a MODELLED product. Every model card says so. What no number in the
repository has said until now is HOW WRONG that stand-in is, because the
comparison needs observed discharge and India does not publish a recent daily
observed series.

The CWC Advisory Flood Forecast portal does publish a rolling ~2-day window of
hourly observed level plus CWC's own derived discharge, and this repository has
been archiving it since 2026-08-26. That accumulating archive is the first
observed yardstick available, so this script measures the modelled river
against it, per station:

    ratio    = mean(GloFAS) / mean(CWC)     1.0 = same size, 0.5 = half
    bias     = mean(GloFAS) - mean(CWC)     m3/s
    corr     = Pearson r on daily means     does it track the ups and downs?
    rel_mae  = MAE / mean(CWC)              typical daily miss, as a fraction

HONESTY, three ways:
- CWC's discharge is itself DERIVED from observed level through a rating curve
  the agency does not publish. It is the official figure and far closer to the
  river than GloFAS, but it is not a direct measurement, and it is labelled that
  way everywhere below.
- The archive is short. Every row carries n_days, and any station with fewer
  than MIN_DAYS is reported as insufficient rather than scored. Nothing here is
  a verdict yet; it is a measurement that gets stronger every 3 hours.
- A ratio far from 1.0 does NOT by itself invalidate a basin's colours. The
  whole chain is GloFAS-relative: models learn GloFAS, thresholds are GloFAS
  percentiles, and the colour compares a GloFAS forecast against them, so a
  roughly constant scale factor cancels. It matters when it is not constant,
  which is what corr and rel_mae are here to expose.

Writes data/history/glofas_vs_observed.json, public/glofas_vs_observed.json
and docs/GLOFAS-VS-OBSERVED.md (+ served copy). Runs every pipeline pass.
"""

import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (DATA_DIR, OBSERVED, PUBLIC_DIR, REPO_ROOT,  # noqa: E402
                    fetch_json, load_json, save_json, utc_now_iso)

HIST = DATA_DIR / "history"
OBS_DIR = HIST / "cwc_aff" / "observed"
DOCS = REPO_ROOT / "docs"
FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"

MIN_DAYS = 5          # below this, report insufficient rather than a number
MIN_DAYS_CORR = 7     # correlation on fewer days than this is noise
IMPLAUSIBLE_HIGH = 5.0   # beyond this the two series are not the same river
IMPLAUSIBLE_LOW = 0.2

SRC_OBS = ("Central Water Commission (CWC), Ministry of Jal Shakti - Advisory Flood "
           "Forecast portal (aff.india-water.gov.in): CWC-derived discharge from "
           "observed WIMS gauge level (rating curves are classified, so this is the "
           "agency's own figure, not a direct measurement)")
SRC_MOD = ("GloFAS v4 consolidated reanalysis via Open-Meteo Flood API at the snapped "
           "river-network cell used by this project's models (MODELLED product)")


def safe(name: str) -> str:
    """Archive filenames are sanitised station names - must match fetch_cwc_aff.safe.

    Six stations ('GUWAHATI(D.C.COURT)', 'MANAS N H CROSSING', ...) contain spaces,
    dots or hyphens, so matching on the raw name silently drops them.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def observed_daily(station: str) -> pd.Series:
    """Daily mean of CWC's published discharge for one station, IST days."""
    df = pd.read_parquet(OBS_DIR / f"{safe(station)}.parquet")
    if "cwc_discharge_m3s" not in df.columns:
        return pd.Series(dtype=float)
    df = df.dropna(subset=["cwc_discharge_m3s"])
    if df.empty:
        return pd.Series(dtype=float)
    day = pd.to_datetime(df["time_ist"]).dt.date
    return df.groupby(day)["cwc_discharge_m3s"].mean()


def glofas_daily(lat: float, lon: float, start: str, end: str) -> pd.Series:
    """GloFAS reanalysis daily discharge at one cell over the observed window."""
    api = fetch_json(FLOOD_API, {"latitude": lat, "longitude": lon,
                                 "daily": "river_discharge",
                                 "start_date": start, "end_date": end})
    d = api.get("daily", {})
    if not d.get("time"):
        return pd.Series(dtype=float)
    return pd.Series(d["river_discharge"],
                     index=pd.to_datetime(d["time"]).date).dropna()


def score(obs: pd.Series, mod: pd.Series) -> dict:
    j = pd.DataFrame({"o": obs, "m": mod}).dropna()
    n = len(j)
    if n < MIN_DAYS:
        return {"n_days": n, "insufficient": True,
                "note": f"only {n} overlapping day(s); need {MIN_DAYS}"}
    o, m = j["o"].to_numpy(), j["m"].to_numpy()
    mo = float(np.mean(o))
    out = {"n_days": n, "insufficient": False,
           "observed_mean_m3s": round(mo, 1),
           "glofas_mean_m3s": round(float(np.mean(m)), 1),
           "ratio_glofas_over_observed": round(float(np.mean(m)) / mo, 3) if mo else None,
           "bias_m3s": round(float(np.mean(m) - mo), 1),
           "mae_m3s": round(float(np.mean(np.abs(m - o))), 1),
           "rel_mae": round(float(np.mean(np.abs(m - o))) / mo, 3) if mo else None}
    if n >= MIN_DAYS_CORR and j["o"].std() > 0 and j["m"].std() > 0:
        out["corr"] = round(float(j["o"].corr(j["m"])), 3)
    else:
        out["corr"] = None
        out["corr_note"] = f"needs {MIN_DAYS_CORR} days and variation in both series"
    # An order-of-magnitude gap is not a measurement of model error - it means the
    # two series are not describing the same water (wrong channel, different units,
    # or a station reporting a local offtake). Flag it for review instead of
    # letting it be read as "GloFAS is 75x too big".
    r = out["ratio_glofas_over_observed"]
    if r is not None and (r > IMPLAUSIBLE_HIGH or r < IMPLAUSIBLE_LOW):
        out["implausible"] = True
        out["implausible_note"] = (
            f"ratio {r} is beyond {IMPLAUSIBLE_LOW}-{IMPLAUSIBLE_HIGH}x: treat as a "
            "station/cell mismatch to investigate, NOT as measured model error")
    return out


def main() -> int:
    now = utc_now_iso()
    try:
        cwc = load_json(PUBLIC_DIR / "cwc_stations.json")["stations"]
        cells = {p["id"]: p for p in load_json(DATA_DIR / "discharge.json")["points"]}
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"glofas_vs_observed: inputs not ready ({e}) - skipped")
        return 0

    rows = []
    for s in cwc:
        rid = s.get("poc_river")
        station = s.get("aff_station")
        if not station:
            continue
        f = OBS_DIR / f"{safe(station)}.parquet"
        if not f.exists():
            rows.append({"station": station, "river": s.get("river"), "poc_river": rid,
                         "n_days": 0, "insufficient": True,
                         "note": "no observed archive file for this station yet"})
            continue
        obs = observed_daily(station)
        if obs.empty:
            rows.append({"station": station, "river": s.get("river"), "poc_river": rid,
                         "n_days": 0, "insufficient": True,
                         "note": "no CWC discharge published for this station yet"})
            continue
        cell = cells.get(rid) if rid else None
        if not cell or cell.get("grid_lat") is None:
            rows.append({"station": station, "river": s.get("river"), "poc_river": rid,
                         "n_days": int(len(obs)), "insufficient": True,
                         "note": "no modelled cell mapped to this station"})
            continue
        try:
            mod = glofas_daily(cell["grid_lat"], cell["grid_lon"],
                               str(obs.index.min()), str(obs.index.max()))
        except Exception as e:                          # network/API, keep going
            rows.append({"station": station, "river": s.get("river"), "poc_river": rid,
                         "n_days": int(len(obs)), "insufficient": True,
                         "note": f"GloFAS fetch failed: {type(e).__name__}"})
            continue
        r = {"station": station, "river": s.get("river"), "poc_river": rid,
             "glofas_cell": [cell["grid_lat"], cell["grid_lon"]],
             "window_ist": [str(obs.index.min()), str(obs.index.max())]}
        r.update(score(obs, mod))
        rows.append(r)

    scored = [r for r in rows if not r.get("insufficient")]
    ok = [r for r in scored if not r.get("implausible")]
    ratios = [r["ratio_glofas_over_observed"] for r in ok
              if r.get("ratio_glofas_over_observed")]
    corrs = [r["corr"] for r in ok if r.get("corr") is not None]
    summary = {
        "stations_total": len(rows),
        "stations_scored": len(scored),
        "stations_in_medians": len(ok),
        "stations_flagged_implausible": len(scored) - len(ok),
        "median_ratio_glofas_over_observed": round(float(np.median(ratios)), 3) if ratios else None,
        "median_corr": round(float(np.median(corrs)), 3) if corrs else None,
        "stations_glofas_below_observed": sum(1 for r in ratios if r < 1.0),
        "stations_glofas_above_observed": sum(1 for r in ratios if r >= 1.0),
    }

    doc = {
        "generated_at": now,
        "what_this_is": ("How far this project's modelled river (GloFAS v4) sits from "
                         "the official CWC figure at the same station. The models are "
                         "trained and scored on GloFAS; this is the first measurement "
                         "of how good a stand-in that is."),
        "observed_source": {"class": OBSERVED, "source": SRC_OBS, "retrieved_at": now},
        "modelled_source": {"class": "SIMULATED", "source": SRC_MOD, "retrieved_at": now},
        "caveats": [
            "CWC discharge is derived from observed level via a rating curve the "
            "agency does not publish - official, but not a direct measurement.",
            "The observed archive begins 2026-08-26 and grows every 3 hours; short "
            "windows are reported as insufficient rather than scored.",
            "A ratio far from 1.0 does not by itself invalidate a basin's colours: "
            "the model, its thresholds and its decision rule are all GloFAS-relative, "
            "so a roughly constant scale factor cancels. Watch corr and rel_mae.",
            "All windows so far are monsoon-season only; low-flow behaviour is untested.",
        ],
        "summary": summary,
        "stations": sorted(rows, key=lambda r: (r.get("insufficient", False),
                                                r.get("station") or "")),
    }
    save_json(HIST / "glofas_vs_observed.json", doc)
    save_json(PUBLIC_DIR / "glofas_vs_observed.json", doc)
    write_md(doc)
    if scored:
        print(f"OK glofas_vs_observed: {len(scored)}/{len(rows)} stations scored, "
              f"median GloFAS/observed ratio {summary['median_ratio_glofas_over_observed']}, "
              f"median corr {summary['median_corr']}")
    else:
        print(f"OK glofas_vs_observed: 0/{len(rows)} stations scored yet "
              f"(archive still shorter than {MIN_DAYS} days)")
    return 0


def write_md(doc: dict) -> None:
    L = ["# GloFAS vs OBSERVED - how far the modelled river is from the real one", ""]
    A = L.append
    A("> Every model here is trained and scored against **GloFAS v4**, a modelled")
    A("> product, not a river gauge. Every model card says so. This page is the")
    A("> first attempt to say *how wrong* that stand-in is, using the CWC observed")
    A("> archive this repository has been accumulating since 2026-08-26.")
    A("")
    A(f"Generated {doc['generated_at']} by `backend/glofas_vs_observed.py`.")
    A("")
    s = doc["summary"]
    A(f"**{s['stations_scored']} of {s['stations_total']} stations scored.** "
      f"Median GloFAS/observed ratio **{s['median_ratio_glofas_over_observed']}**, "
      f"median correlation **{s['median_corr']}**. "
      f"GloFAS reads lower than CWC at {s['stations_glofas_below_observed']} station(s), "
      f"higher at {s['stations_glofas_above_observed']}. "
      f"{s['stations_flagged_implausible']} station(s) flagged as a likely station/cell "
      f"mismatch and excluded from the medians.")
    A("")
    A("| Station | River | n days | CWC mean | GloFAS mean | ratio | bias | corr | rel MAE |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in doc["stations"]:
        if r.get("insufficient"):
            A(f"| {r['station']} | {r.get('river') or '-'} | {r.get('n_days', 0)} | "
              f"insufficient | | | | | |")
            continue
        flag = " **(mismatch - see note)**" if r.get("implausible") else ""
        A(f"| {r['station']}{flag} | {r.get('river') or '-'} | {r['n_days']} | "
          f"{r['observed_mean_m3s']} | {r['glofas_mean_m3s']} | "
          f"{r['ratio_glofas_over_observed']} | {r['bias_m3s']} | "
          f"{r['corr'] if r['corr'] is not None else '-'} | {r['rel_mae']} |")
    A("")
    A("All discharge in m3/s. `ratio` is GloFAS mean over CWC mean: 1.0 means the")
    A("same size river, 0.5 means GloFAS carries half the water. `corr` is on daily")
    A("means and answers a different question - whether GloFAS rises and falls with")
    A("the real river even when it has the size wrong.")
    A("")
    A("## How to read this honestly")
    A("")
    for c in doc["caveats"]:
        A(f"- {c}")
    A("")
    A("## Sources")
    A("")
    A(f"- Observed: {doc['observed_source']['source']}")
    A(f"- Modelled: {doc['modelled_source']['source']}")
    md = "\n".join(L) + "\n"
    DOCS.mkdir(exist_ok=True)
    (DOCS / "GLOFAS-VS-OBSERVED.md").write_text(md, encoding="utf-8")
    (PUBLIC_DIR / "GLOFAS-VS-OBSERVED.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
