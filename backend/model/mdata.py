"""Shared data/feature/metric utilities for the modelling recipe (generalised
to basins in Phase 1; Dikhow remains the default everywhere).

Loads the history partitions into daily frames, defines the walk-forward
folds, the feature builder used by baselines/training/inference (one code
path = no train/serve skew), and the metric definitions.

HONESTY: the target is GloFAS v4 reanalysis discharge - a MODELLED product,
not observed river data; observed CWC gauge data will replace it when access
is granted. GloFAS is daily -> horizons are 1 and 2 days.

Leakage rule: every feature at prediction day d uses ONLY data with
timestamp <= end of day d. leakage_test() verifies this mechanically by
corrupting the future and asserting features at <= d are unchanged.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from basins import BASINS, rain_point_ids  # noqa: E402
from lag_analysis import (HIST_DIR,  # noqa: E402
                          daily_rain_sums, load_daily_q, load_hourly_rain)

YEARS_ALL = list(range(2015, 2026))
UPSTREAM_IDS = [u["id"] for u in BASINS["dikhow"]["upstream"]]
HORIZONS = [1, 2]                 # days (GloFAS is daily; 6/12 h impossible)
PRIMARY_HORIZON = 1
# walk-forward: train years -> held-out validation year (monsoon Jun-Sep)
FOLDS = [(list(range(2015, 2022)), 2022),
         (list(range(2015, 2023)), 2023),
         (list(range(2015, 2024)), 2024),
         (list(range(2015, 2025)), 2025)]

FEATURES = ["q", "q_lag1", "q_lag2", "dq", "dq_lag1", "trend48_pct",
            "rain_6h", "rain_12h", "rain_24h", "rain_48h",
            "rain_24h_lag1", "rain_14d", "rain_30d",
            "doy_sin", "doy_cos"]
# upstream q features appended dynamically if partitions exist


def upstream_available(basin: str = "dikhow") -> list[str]:
    return [u["id"] for u in BASINS[basin]["upstream"]
            if (HIST_DIR / "discharge" / u["id"]).exists()
            and any((HIST_DIR / "discharge" / u["id"]).glob("*.parquet"))]


def build_daily_frame(years, basin: str = "dikhow") -> pd.DataFrame:
    """Daily frame: target q, rain windows, antecedent sums, upstream q."""
    rain_h = load_hourly_rain(years, rain_point_ids(basin))
    q = load_daily_q(BASINS[basin]["target"], years)
    df = daily_rain_sums(rain_h)                      # rain_6h..rain_48h
    daily_tot = rain_h.groupby(rain_h.index.date).sum()
    daily_tot.index = pd.to_datetime(daily_tot.index)
    df["rain_14d"] = daily_tot.rolling(14, min_periods=10).sum()
    df["rain_30d"] = daily_tot.rolling(30, min_periods=21).sum()
    df["q"] = q
    for pid in upstream_available(basin):
        qu = load_daily_q(pid, years)
        df[f"qup_{pid}"] = qu
        df[f"dqup_{pid}"] = qu.diff()
    return df.sort_index()


def feature_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Features at day d (only data <= d) + targets y_h1/y_h2 (future)."""
    f = pd.DataFrame(index=daily.index)
    f["q"] = daily["q"]
    f["q_lag1"] = daily["q"].shift(1)
    f["q_lag2"] = daily["q"].shift(2)
    f["dq"] = daily["q"].diff()
    f["dq_lag1"] = f["dq"].shift(1)
    f["trend48_pct"] = 100.0 * (daily["q"] - daily["q"].shift(2)) / daily["q"].shift(2)
    for col in ("rain_6h", "rain_12h", "rain_24h", "rain_48h",
                "rain_14d", "rain_30d"):
        f[col] = daily[col]
    f["rain_24h_lag1"] = daily["rain_24h"].shift(1)
    doy = f.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # upstream columns in the frame's own insertion order (qup_x, dqup_x, ...)
    for col in daily.columns:
        if col.startswith("qup_") or col.startswith("dqup_"):
            f[col] = daily[col]
    for h in HORIZONS:
        f[f"y_h{h}"] = daily["q"].shift(-h)           # target: FUTURE only
    return f


def feature_cols(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if not c.startswith("y_h")]


def leakage_test(daily: pd.DataFrame, probe_day=None) -> None:
    """Corrupt everything AFTER probe_day; features at <= probe_day must be
    bit-identical (targets excluded - they are labels, not features)."""
    f0 = feature_frame(daily)
    if probe_day is None:
        probe_day = daily.index[int(len(daily) * 0.6)]
    corrupted = daily.copy()
    rng = np.random.default_rng(0)
    after = corrupted.index > probe_day
    for col in corrupted.columns:
        corrupted.loc[after, col] = rng.uniform(0, 1e6, int(after.sum()))
    f1 = feature_frame(corrupted)
    cols = feature_cols(f0)
    a = f0.loc[:probe_day, cols].to_numpy()
    b = f1.loc[:probe_day, cols].to_numpy()
    if not np.allclose(a, b, equal_nan=True):
        raise AssertionError("LEAKAGE: features at <= t changed when the "
                             "future was corrupted")


def monsoon_mask(index) -> np.ndarray:
    return (index.month >= 6) & (index.month <= 9)


def train_q90(daily: pd.DataFrame, train_years) -> float:
    """90th percentile of TRAINING-years monsoon discharge (no leakage)."""
    q = daily["q"][daily.index.year.isin(train_years) & monsoon_mask(daily.index)]
    return float(np.nanpercentile(q.dropna(), 90))


def mae(a, b) -> float:
    return float(np.nanmean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def event_metrics(actual: pd.Series, alarm_target_days: pd.Series,
                  thr: float, horizon: int) -> dict:
    """Day-basis contingency + event lead time.
    POD = predicted-exceedance days / actual-exceedance days;
    FAR = false-alarm days / all alarm days;
    lead(event) = onset_day - first prediction day whose target day is any
    day of that event and predicted >= thr (persistence therefore scores 0;
    negative = alarm only after onset)."""
    act = actual >= thr
    pred = alarm_target_days >= thr
    both = act & pred
    n_act, n_pred = int(act.sum()), int(pred.sum())
    pod = float(both.sum() / n_act) if n_act else None
    far = float((pred & ~act).sum() / n_pred) if n_pred else None
    leads = []
    onsets = []
    prev = False
    for d, is_ex in act.items():
        if is_ex and not prev:
            onsets.append(d)
        prev = is_ex
    for o in onsets:
        run = []
        d = o
        while d in act.index and act[d]:
            run.append(d)
            d += pd.Timedelta(days=1)
        alarm_days = [dd - pd.Timedelta(days=horizon) for dd in run
                      if dd in pred.index and pred[dd]]
        if alarm_days:
            leads.append((o - min(alarm_days)).days)
    return {"threshold_m3s": round(thr, 1), "n_exceed_days": n_act,
            "n_alarm_days": n_pred, "n_events": len(onsets),
            "n_detected": len(leads),
            "pod_days": round(pod, 3) if pod is not None else None,
            "far_days": round(far, 3) if far is not None else None,
            "mean_lead_days": round(float(np.mean(leads)), 2) if leads else None}


def season_metrics(actual: pd.Series, pred: pd.Series, pers: pd.Series,
                   thr: float, horizon: int) -> dict:
    """All metrics for one held-out monsoon season, one horizon."""
    df = pd.DataFrame({"a": actual, "p": pred, "pers": pers}).dropna()
    out = {"n_days": int(len(df)),
           "mae": round(mae(df["a"], df["p"]), 1),
           "rmse": round(rmse(df["a"], df["p"]), 1)}
    mae_pers = mae(df["a"], df["pers"])
    out["mae_persistence"] = round(mae_pers, 1)
    out["skill_vs_persistence"] = round(1.0 - out["mae"] / mae_pers, 3) if mae_pers else None
    out.update(event_metrics(df["a"], df["p"], thr, horizon))
    return out
