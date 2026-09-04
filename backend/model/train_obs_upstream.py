"""OU2: observed-upstream experiment - fixed in advance of training.

Does REAL observed upstream discharge (CWC Arunachal telemetry via NWDP)
improve the Jia Bharali model over its shipped v0 feature set?

Everything judgeable was fixed before training (see rules fixed before training):
pair = jiabharali + tenga_1; variants V-same (diagnostic, unshippable)
and V-lag3 (shippable, matches the feed's ~2-3 day latency); ship rule =
V-lag3 beats fold-matched v0 on monsoon MAE h=1 in BOTH 2024 and 2025;
2026 fold informative only. Target remains GloFAS reanalysis - a
modelled product, not observed river data (unchanged caveat).
"""
from __future__ import annotations
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdata import build_daily_frame, feature_cols, feature_frame, monsoon_mask  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OBS_PARQUET = (BASE_DIR / "data" / "history" / "nwdp" /
               "cwc_arunachal_discharge" / "tenga_1" / "data.parquet")
OUT_JSON = BASE_DIR / "data" / "history" / "obs_upstream_experiment.json"
OUT_MD = BASE_DIR / "docs" / "OBS-UPSTREAM-EXPERIMENT.md"

BASIN = "jiabharali"
STATION = "tenga_1"
HGB_PARAMS = {"max_iter": 400, "learning_rate": 0.05, "random_state": 0}
FOLD_YEARS = (2024, 2025)
MIN_COVERAGE = 0.8  # fraction of monsoon test days with obs present


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def obs_daily() -> pd.Series:
    df = pd.read_parquet(OBS_PARQUET)
    t = pd.to_datetime(df["t"]) - pd.Timedelta(hours=5, minutes=30)  # IST -> UTC
    v = pd.to_numeric(df["value"], errors="coerce")
    s = pd.Series(v.values, index=t).dropna()
    s = s[(s >= 0) & (s < 1e6)]
    d = s.groupby(s.index.date).mean()
    d.index = pd.to_datetime(d.index)
    return d.sort_index()


def add_obs_features(feats: pd.DataFrame, obs: pd.Series) -> pd.DataFrame:
    """Both variants' columns; NaN where the station has no data (HGB-native)."""
    f = feats.copy()
    o = obs.reindex(f.index)
    f["obsq"] = o                     # V-same (diagnostic)
    f["obsq_lag1"] = o.shift(1)
    f["dobsq"] = o.diff()
    f["obsq_lag3"] = o.shift(3)      # V-lag3 (shippable parity)
    f["obsq_lag4"] = o.shift(4)
    f["dobsq_lag3"] = o.diff().shift(3)
    return f


def leakage_check(feats: pd.DataFrame, obs: pd.Series) -> None:
    """Corrupt obs AFTER a probe day; features up to the probe day must not move."""
    probe = pd.Timestamp("2025-07-15")
    corrupted = obs.copy()
    corrupted[corrupted.index > probe] = 9.9e5
    a = add_obs_features(feats, obs).loc[:probe]
    b = add_obs_features(feats, corrupted).loc[:probe]
    if not a.fillna(-1).equals(b.fillna(-1)):
        raise SystemExit("LEAKAGE: obs features changed when the future was corrupted")
    print("  obs leakage test PASSED (corrupt-the-future)")


def fold_mae(feats: pd.DataFrame, cols: list[str], train_years, test_lo, test_hi):
    """Fit h=1 on train_years, return monsoon MAE + n_days on [test_lo, test_hi]."""
    y = feats["q"].shift(-1)          # tomorrow's discharge
    ok = y.notna()
    tr = ok & feats.index.year.isin(train_years)
    te = (ok & (feats.index >= test_lo) & (feats.index <= test_hi)
          & monsoon_mask(feats.index))
    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(feats.loc[tr, cols], y[tr])
    pred = pd.Series(model.predict(feats.loc[te, cols]), index=feats.index[te])
    err = float((y[te] - pred).abs().mean())
    return err, int(te.sum())


def main() -> int:
    now = utc_now_iso()
    print(f"[{BASIN}] observed-upstream experiment ({STATION}); rules: rules fixed before training")
    years = list(range(2015, 2027))
    daily = build_daily_frame(years, BASIN)
    feats0 = feature_frame(daily)
    obs = obs_daily()
    leakage_check(feats0, obs)
    feats = add_obs_features(feats0, obs)
    v0_cols = feature_cols(feats0)
    vsame_cols = v0_cols + ["obsq", "obsq_lag1", "dobsq"]
    vlag3_cols = v0_cols + ["obsq_lag3", "obsq_lag4", "dobsq_lag3"]

    folds = [([y for y in range(2015, fy)], f"{fy}-06-01", f"{fy}-09-30", str(fy))
             for fy in FOLD_YEARS]
    folds.append((list(range(2015, 2026)), "2026-06-01", "2026-08-31",
                  "2026 (informative only)"))

    rows = []
    for train_years, lo, hi, label in folds:
        mon = feats[(feats.index >= lo) & (feats.index <= hi)]
        cov = float(mon["obsq"].notna().mean()) if len(mon) else 0.0
        r = {"fold": label, "obs_coverage": round(cov, 3)}
        for name, cols in (("v0", v0_cols), ("V-same", vsame_cols), ("V-lag3", vlag3_cols)):
            m, n = fold_mae(feats, cols, train_years, lo, hi)
            r[name] = round(m, 1); r["n_days"] = n
        rows.append(r)
        print(f"  fold {label}: cov={cov:.0%} n={r['n_days']} | "
              f"v0={r['v0']} V-same={r['V-same']} V-lag3={r['V-lag3']}")

    # ---- verdict per pre-registered rule ----
    primary = rows[:2]
    coverage_ok = all(r["obs_coverage"] >= MIN_COVERAGE for r in primary)
    lag3_wins = sum(1 for r in primary if r["V-lag3"] < r["v0"])
    same_wins = sum(1 for r in primary if r["V-same"] < r["v0"])
    ship = coverage_ok and lag3_wins == 2
    fold26 = rows[2]
    promising = (not ship) and (fold26["V-lag3"] < fold26["v0"])
    if ship:
        verdict = ("SHIP earned: V-lag3 beat the v0 feature set in both held-out "
                   "monsoons - wiring as jiabharali v1-obs is the next step")
    elif promising:
        verdict = ("NOT earned on 2024+2025 (%d/2), but V-lag3 wins the informative "
                   "2026 fold - pre-registered follow-up: re-judge after 2026-09-30 "
                   "with rule 'beat v0 in BOTH 2025 and 2026'" % lag3_wins)
    else:
        verdict = ("NOT earned: V-lag3 beat v0 in %d/2 held-out monsoons (2026 fold "
                   "also no win). Shipped v0 stays. The observed record is simply "
                   "young; re-run when it grows" % lag3_wins)
    print(f"  DECISION: {verdict}")

    OUT_JSON.write_text(json.dumps({
        "generated_at": now, "class": "OBSERVED",
        "source": ("experiment vs CWC observed telemetry discharge at Tenga "
                   "(NWDP open data); target GloFAS v4 reanalysis - modelled, "
                   "not observed river data"),
        "basin": BASIN, "station": STATION, "hgb_params": HGB_PARAMS,
        "rule": "V-lag3 must beat v0 monsoon MAE h=1 in BOTH 2024 and 2025",
        "folds": rows, "same_wins": same_wins, "lag3_wins": lag3_wins,
        "coverage_ok": coverage_ok, "ship": ship, "verdict": verdict,
    }, indent=1), encoding="utf-8")

    md = [
        "# OBS-UPSTREAM-EXPERIMENT - real measured water as a model input\n",
        f"Generated {now} by `backend/model/train_obs_upstream.py`. Rules were",
        "fixed in advance of training and committed before this ran.",
        "**Question:** the Brahmaputra models won by watching modelled water",
        "already in the river upstream. Does REAL measured water - CWC's",
        "telemetry gauge on the upper Kameng at Tenga - improve the Jia Bharali",
        "model beyond its shipped v0?\n",
        "**The latency catch, stated first:** NWDP publishes observations ~2-3",
        "days late, so a live model can only use LAGGED observations. V-same",
        "(same-day) is diagnostic only and can never ship; V-lag3 is the only",
        "shippable variant.\n",
        "_Target caveat unchanged: models are trained and judged against GloFAS",
        "reanalysis - a modelled product, not observed river data._\n",
        "## Held-out monsoon MAE at h=1 (m3/s; lower is better)\n",
        "| fold | obs coverage | days | v0 | V-same | V-lag3 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['fold']} | {r['obs_coverage']:.0%} | {r['n_days']} "
                  f"| {r['v0']} | {r['V-same']} | {r['V-lag3']} |")
    md += [f"\n**Verdict (pre-registered rule): {verdict}.**\n",
           "Deferred by coverage (also pre-registered): disang (Kanubari/Tissa)",
           "and the Dibrugarh reach (Aalo/Basar) have no judgeable 2024 monsoon;",
           "both become judgeable on 2025+2026 after 2026-09-30.\n"]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"OK experiment -> {OUT_JSON.name} + docs/{OUT_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
