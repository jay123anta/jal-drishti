"""MODEL v0 (generic per basin since Phase 1c) - live inference.

    python backend/model/predict.py                 # dikhow
    python backend/model/predict.py --basin kopili

Runs the SHIPPED model v0 of the basin (trained_hgb / B3_lag_rule /
B1_persistence - decided by train.py under the pre-registered rule) on
live inputs and writes data/<forecast> (basins.model_names) for
classify_risk.py.

Live inputs (fast, batched):
- one Open-Meteo call: past 35 d hourly rain for the basin's anchor cells
  (forecast product, past hours = model analysis; NOTE: training used the
  ERA5-family archive - a documented source shift);
- target discharge from data/discharge.json (pipeline live fetch);
- upstream cells (if the shipped model uses them): one batched Flood API
  call, past 30 d.
The ML path builds a small daily frame and runs the SAME feature builder
as training (mdata.feature_frame) - one code path, no train/serve skew.

Colour mapping (documented in the model card): P(exceed q90 at 1 day)
>= 0.5 -> RED, >= 0.2 -> YELLOW, else GREEN; P from the shipped model's
held-out residuals.

Degradation: any missing input or fetch failure -> {degraded: true, error}
and exit 0; classify_risk.py keeps the heuristic for that basin, marked.

Target caveat: predictions are modelled discharge against GloFAS v4
reanalysis - NOT observed river levels.
"""

import datetime
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import (DATA_DIR, FORECAST, OBSERVED, REPO_ROOT, SIMULATED,  # noqa: E402
                    fetch_json, load_json, save_failed_request, save_json,
                    utc_now_iso)

from basins import BASINS, model_names  # noqa: E402
from lag_analysis import daily_rain_sums  # noqa: E402
from mdata import feature_frame  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
RAIN_API = "https://api.open-meteo.com/v1/forecast"
FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
CUTOFF_RED, CUTOFF_YELLOW = 0.5, 0.2


def write_degraded(path, basin: str, reason: str, now: str) -> None:
    save_json(path, {"generated_at": now, "basin": basin, "degraded": True, "error": reason,
                     "note": (f"model v0 inference unavailable for {basin} - its villages "
                              f"fall back to the heuristic, marked degraded; nothing fabricated")})
    print(f"  [{basin}] model inference DEGRADED: {reason}")


def basis_sentence(colour: str, river_plain: str, trend_word: str) -> str:
    if colour == "RED":
        return f"A tested river model expects unusually high water on the {river_plain} near here by tomorrow."
    if colour == "YELLOW":
        return (f"A tested river model sees a raised chance of high water on the {river_plain} "
                f"near here by tomorrow - worth watching.")
    return (f"A tested river model expects the {river_plain} near here to stay in its normal "
            f"range through tomorrow (river currently {trend_word}).")


def main() -> int:
    args = sys.argv[1:]
    basin = args[args.index("--basin") + 1] if "--basin" in args else "dikhow"
    if basin not in BASINS:
        raise SystemExit(f"unknown basin {basin!r}")
    cfg, nm = BASINS[basin], model_names(basin)
    out_path = DATA_DIR / nm["forecast"]
    now = utc_now_iso()
    src_model = (f"JalDrishti model v0 for {basin} - shipped: %s, fitted on 2015-2025 GloFAS "
                 f"reanalysis (docs/{nm['card']}); modelled discharge, NOT observed river levels")

    try:
        meta = load_json(MODELS_DIR / nm["meta"])
        thr = meta["threshold_q90_monsoon_2015_2025_m3s"]
        residuals = meta["validation_residuals_m3s"]
        shipped = meta["shipped"]
    except (FileNotFoundError, ValueError, KeyError) as err:
        write_degraded(out_path, basin, f"model meta unavailable: {err}", now)
        return 0
    src_model = src_model % shipped

    # ---- target discharge (pipeline live fetch) ----
    try:
        disch = load_json(DATA_DIR / "discharge.json")
        tp = next(p for p in disch["points"] if p["id"] == cfg["target"])
        obs = [r for r in tp["daily"] if r["class"] == OBSERVED]
        latest = obs[-1]
        q_now = latest["discharge_m3s"]
        trend_word = "steady"
        if len(obs) >= 3 and obs[-3]["discharge_m3s"] > 0:
            pct = 100.0 * (q_now - obs[-3]["discharge_m3s"]) / obs[-3]["discharge_m3s"]
            trend_word = "rising" if pct >= 5 else ("falling" if pct <= -5 else "steady")
    except (FileNotFoundError, ValueError, StopIteration, IndexError, KeyError) as err:
        write_degraded(out_path, basin, f"live discharge unavailable: {err}", now)
        return 0

    # ---- catchment rain, past 35 d hourly, one batched call ----
    pts = cfg["rain_points"]
    params = {"latitude": ",".join(str(p["lat"]) for p in pts),
              "longitude": ",".join(str(p["lon"]) for p in pts),
              "hourly": "precipitation", "past_days": 35, "forecast_days": 1,
              "timezone": "UTC"}
    try:
        api = fetch_json(RAIN_API, params)
    except RuntimeError as err:
        save_failed_request(f"model_catchment_rain_{basin}", RAIN_API, params, str(err))
        write_degraded(out_path, basin, f"live catchment rain fetch failed: {err}", now)
        return 0
    cells = api if isinstance(api, list) else [api]
    if len(cells) != len(pts):
        write_degraded(out_path, basin, f"expected {len(pts)} cells, got {len(cells)}", now)
        return 0
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    series = []
    for c in cells:
        idx = pd.to_datetime(c["hourly"]["time"], utc=True)
        s = pd.Series(c["hourly"]["precipitation"], index=idx, dtype="float")
        series.append(s[idx <= now_dt])
    rain_h = pd.concat(series, axis=1).mean(axis=1).sort_index()
    rain_h.index = rain_h.index.tz_localize(None)
    rain24 = float(rain_h[rain_h.index > rain_h.index[-1] - pd.Timedelta(hours=24)].sum())
    rain48 = float(rain_h[rain_h.index > rain_h.index[-1] - pd.Timedelta(hours=48)].sum())

    # ---- point prediction per horizon ----
    preds = {}
    if shipped == "trained_hgb":
        try:
            bundle = __import__("joblib").load(MODELS_DIR / nm["pkl"])
            df = daily_rain_sums(rain_h)
            tot = rain_h.groupby(rain_h.index.date).sum()
            tot.index = pd.to_datetime(tot.index)
            df["rain_14d"] = tot.rolling(14, min_periods=10).sum()
            df["rain_30d"] = tot.rolling(30, min_periods=21).sum()
            qs = pd.Series({pd.Timestamp(r["date"]): r["discharge_m3s"] for r in obs})
            df["q"] = qs
            up_ids = [u["id"] for u in cfg["upstream"]
                      if any(f.startswith(f"qup_{u['id']}") for f in bundle["features"])]
            if up_ids:
                state = load_json(REPO_ROOT / "BACKFILL-STATE.json")
                cells_up = {p["id"]: p for p in state["river_points_by_basin"][basin]}
                up = [cells_up[i] for i in up_ids]
                uapi = fetch_json(FLOOD_API, {
                    "latitude": ",".join(str(u["lat"]) for u in up),
                    "longitude": ",".join(str(u["lon"]) for u in up),
                    "daily": "river_discharge", "past_days": 30, "forecast_days": 1,
                    "timezone": "UTC"})
                ucells = uapi if isinstance(uapi, list) else [uapi]
                for u, uc in zip(up, ucells):
                    us = pd.Series(uc["daily"]["river_discharge"],
                                   index=pd.to_datetime(uc["daily"]["time"]), dtype="float")
                    us = us[us.index <= pd.Timestamp(now_dt.date())]
                    df[f"qup_{u['id']}"] = us
                    df[f"dqup_{u['id']}"] = us.diff()
            feats = feature_frame(df.sort_index())
            cols = bundle["features"]
            ok = feats.dropna(subset=cols)
            if ok.empty:
                raise RuntimeError("no complete feature row in the live window")
            row = ok.iloc[[-1]][cols]
            for h in (1, 2):
                preds[h] = float(bundle["models"][f"h{h}"].predict(row)[0])
            feature_day = str(ok.index[-1].date())
        except Exception as err:  # noqa: BLE001 - degrade, never fabricate
            write_degraded(out_path, basin, f"ML inference failed: {err}", now)
            return 0
    elif shipped == "B3_lag_rule":
        slopes = meta["b3_slopes_full_period"]
        preds = {1: q_now + slopes["1"] * rain24, 2: q_now + slopes["2"] * rain48}
        feature_day = latest["date"]
    else:  # B1_persistence
        preds = {1: q_now, 2: q_now}
        feature_day = latest["date"]

    out_preds = {}
    for h in (1, 2):
        r = np.asarray(residuals[str(h)])
        out_preds[h] = {"qhat": round(preds[h], 1),
                        "p_exceed": round(float(np.mean(preds[h] + r >= thr)), 3)}
    p1 = out_preds[1]["p_exceed"]
    colour = "RED" if p1 >= CUTOFF_RED else ("YELLOW" if p1 >= CUTOFF_YELLOW else "GREEN")

    def pv(value, cls, source, **extra):
        return {"value": value, "class": cls, "source": source, "retrieved_at": now, **extra}

    river_plain = cfg["label"].split(" (")[0]
    save_json(out_path, {
        "generated_at": now, "degraded": False, "basin": basin,
        "model": nm["method"], "shipped": shipped, "target_caveat": meta["target_caveat"],
        "model_card": f"docs/{nm['card']}", "feature_day": feature_day,
        "inputs": {
            "q_today_m3s": pv(q_now, latest["class"], latest["source"], date=latest["date"]),
            "catchment_rain_24h_mm": pv(round(rain24, 1), OBSERVED,
                                        f"Open-Meteo forecast product past hours (model analysis), "
                                        f"mean of {len(pts)} {basin} catchment cells (CATCHMENT.md)"),
            "catchment_rain_48h_mm": pv(round(rain48, 1), OBSERVED,
                                        f"Open-Meteo forecast product past hours (model analysis), "
                                        f"mean of {len(pts)} {basin} catchment cells (CATCHMENT.md)"),
        },
        "threshold_m3s": pv(thr, OBSERVED, "90th percentile of 2015-2025 monsoon GloFAS "
                                            "reanalysis (NOT an official danger level)"),
        "predictions": {f"h{h}": {
            "horizon": f"{h} day{'s' if h > 1 else ''}",
            "q_m3s": pv(out_preds[h]["qhat"], FORECAST, src_model),
            "p_exceed_threshold": pv(out_preds[h]["p_exceed"], FORECAST,
                                     src_model + " - probability from held-out validation residuals"),
        } for h in (1, 2)},
        "colour": pv(colour, SIMULATED,
                     f"decision rule on model v0 output: P(exceed) >= {CUTOFF_RED} RED, "
                     f">= {CUTOFF_YELLOW} YELLOW, else GREEN (documented cutoffs, arbitrary)",
                     p_exceed_h1=p1),
        "forecast_horizon": "1 day",
        "basis_plain": basis_sentence(colour, river_plain, trend_word),
    })
    print(f"  [{basin}] {shipped}: q_now={q_now:.0f} rain24={rain24:.1f}mm -> h1 "
          f"{out_preds[1]['qhat']:.0f} m3/s (P_exceed {p1:.2f}) -> {colour}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
