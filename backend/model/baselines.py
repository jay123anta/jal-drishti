"""MODEL v0, STEP M3 - mandatory baselines for the Dikhow discharge model.

B1 persistence:  qhat(d+h) = q(d)
B2 climatology:  qhat(d+h) = mean q on that day-of-year (+/- 7 d window)
                 over the TRAINING years only
B3 lag-rule:     qhat(d+h) = q(d) + slope_h * lagged catchment rain
                 (the heuristic done properly: Step M2's lag says tomorrow's
                 change follows today's rain; slope fitted on TRAINING
                 monsoons only, by least squares on dq)

Horizons: 1 day and 2 days. GloFAS discharge is DAILY - 6 h / 12 h horizons
do not exist against this target and are not reported.

Evaluation: walk-forward folds (train 2015-2021 -> monsoon 2022; extend ->
2023; -> 2024; -> 2025). Metrics per held-out monsoon season, never on
training years, never blended: MAE, RMSE, skill vs persistence, and
event-detection stats (POD, FAR, mean lead) against the TRAINING-years
monsoon 90th-percentile threshold.

Target caveat on every artifact: GloFAS reanalysis is a MODELLED product,
not observed river data; observed CWC gauge data will replace it when
access is granted.

Writes data/history/baseline_metrics.json (consumed by the model card).
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import load_json, save_json, utc_now_iso  # noqa: E402

from mdata import (FOLDS, HORIZONS, HIST_DIR, build_daily_frame,  # noqa: E402
                   monsoon_mask, season_metrics, train_q90)


def b1_persistence(daily: pd.DataFrame, h: int) -> pd.Series:
    """Prediction FOR day d, made at day d-h."""
    return daily["q"].shift(h)


def b2_climatology(daily: pd.DataFrame, train_years, h: int) -> pd.Series:
    train = daily[daily.index.year.isin(train_years)]
    doy_mean = {}
    tdoy = train.index.dayofyear.to_numpy()
    tq = train["q"].to_numpy()
    for doy in range(1, 367):
        d = np.abs(tdoy - doy)
        d = np.minimum(d, 366 - d)                    # wrap around new year
        sel = tq[d <= 7]
        doy_mean[doy] = float(np.nanmean(sel)) if len(sel) else np.nan
    return pd.Series([doy_mean[d] for d in daily.index.dayofyear],
                     index=daily.index)


def fit_b3_slopes(daily: pd.DataFrame, train_years, lag_days: int) -> dict:
    """Least-squares slope of dq over h days vs the rain sum available at
    prediction time, TRAINING monsoons only."""
    slopes = {}
    train = daily[daily.index.year.isin(train_years) & monsoon_mask(daily.index)]
    for h in HORIZONS:
        rain_col = "rain_24h" if h == 1 else "rain_48h"
        x = train[rain_col]
        y = train["q"].shift(-h) - train["q"]          # future change (labels)
        df = pd.DataFrame({"x": x, "y": y}).dropna()
        denom = float((df["x"] ** 2).sum())
        slopes[h] = float((df["x"] * df["y"]).sum() / denom) if denom else 0.0
    return slopes


def b3_lag_rule(daily: pd.DataFrame, slopes: dict, h: int) -> pd.Series:
    """Prediction FOR day d made at d-h: q(d-h) + slope * rain seen by d-h."""
    rain_col = "rain_24h" if h == 1 else "rain_48h"
    return daily["q"].shift(h) + slopes[h] * daily[rain_col].shift(h)


def main() -> int:
    now = utc_now_iso()
    args = sys.argv[1:]
    basin = args[args.index("--basin") + 1] if "--basin" in args else "dikhow"
    lag_file = "lag_summary.json" if basin == "dikhow" else f"lag_summary_{basin}.json"
    out_file = "baseline_metrics.json" if basin == "dikhow" else f"baseline_metrics_{basin}.json"
    lag = load_json(HIST_DIR / lag_file)
    lag_days = lag["median_event_lag_days"]
    daily = build_daily_frame(list(range(2015, 2026)), basin)

    results = {"generated_at": now, "basin": basin,
               "target_caveat": ("evaluated against GloFAS v4 reanalysis - a "
                                 "modelled product, not observed river data; "
                                 "observed CWC gauge data will replace this "
                                 "when access is granted"),
               "lag_days_used_by_b3": lag_days,
               "horizons_days": HORIZONS,
               "horizon_note": ("GloFAS is daily; 6/12 h horizons do not "
                                "exist against this target"),
               "folds": []}

    for train_years, val_year in FOLDS:
        thr = train_q90(daily, train_years)
        slopes = fit_b3_slopes(daily, train_years, lag_days)
        val_mask = (daily.index.year == val_year) & monsoon_mask(daily.index)
        actual = daily["q"][val_mask]
        fold = {"train_years": f"{train_years[0]}-{train_years[-1]}",
                "validation_season": f"monsoon {val_year}",
                "threshold_q90_train_m3s": round(thr, 1),
                "b3_slopes": {str(h): round(s, 4) for h, s in slopes.items()},
                "baselines": {}}
        for h in HORIZONS:
            pers = b1_persistence(daily, h)[val_mask]
            clim = b2_climatology(daily, train_years, h)[val_mask]
            lagr = b3_lag_rule(daily, slopes, h)[val_mask]
            fold["baselines"][f"h{h}"] = {
                "B1_persistence": season_metrics(actual, pers, pers, thr, h),
                "B2_climatology": season_metrics(actual, clim, pers, thr, h),
                "B3_lag_rule": season_metrics(actual, lagr, pers, thr, h),
            }
        results["folds"].append(fold)
        b3m = fold["baselines"]["h1"]["B3_lag_rule"]
        print(f"  {fold['validation_season']}: thr={thr:.0f} m3/s | h1 MAE "
              f"pers={fold['baselines']['h1']['B1_persistence']['mae']} "
              f"clim={fold['baselines']['h1']['B2_climatology']['mae']} "
              f"lag={b3m['mae']} (skill {b3m['skill_vs_persistence']}, "
              f"POD {b3m['pod_days']}, FAR {b3m['far_days']}, "
              f"lead {b3m['mean_lead_days']} d)")

    save_json(HIST_DIR / out_file, results)

    # ---- verification ----
    check = load_json(HIST_DIR / out_file)
    assert len(check["folds"]) == len(FOLDS)
    for fold in check["folds"]:
        for h in HORIZONS:
            for b in ("B1_persistence", "B2_climatology", "B3_lag_rule"):
                m = fold["baselines"][f"h{h}"][b]
                assert m["n_days"] > 100, f"{fold['validation_season']} {b}: too few days"
                assert m["mae"] is not None
    print(f"OK data/history/{out_file}: {len(FOLDS)} folds x "
          f"{len(HORIZONS)} horizons x 3 baselines, held-out monsoons only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
