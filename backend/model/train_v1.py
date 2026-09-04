"""MODEL v1, STEP V1b - train + validate with the forecast-rain feature.

Design (pre-registered in internal records / DECISIONS BEFORE this ran):
- PERFECT-PROG TRAINING: candidates learn the rain-response using the
  ACTUAL next-day catchment rain as the feature (2015 - train-end).
- OPERATIONAL VALIDATION: held-out monsoons are scored feeding the REAL
  ISSUED forecasts (Previous Runs archive; exists Jun 2024+), so forecast
  error hits the metrics honestly. Folds: 2015-2023 -> monsoon 2024,
  2015-2024 -> monsoon 2025.
- Candidates at h=1: B4 (lag rule + fc feature, slope fit on training),
  v1-HGB, v1-XGB (inner-validation on the last training monsoon selects
  which ML is gated; both reported). B3/v0 recomputed for reference.
- ANALYSIS-ONLY upper bound: the gated ML fed ACTUAL rain at validation
  (what a perfect rain forecast would buy). Never shippable.

SHIP RULE (fixed in the kickoff commit): v1-ML ships only if it beats
BOTH B4 and B3 in BOTH covered seasons on monsoon MAE at h=1; else B4
ships if it beats B3 in both; else v0 stays.

Target caveat: trained and evaluated against GloFAS v4 reanalysis - a
MODELLED product, not observed river data; observed CWC gauge data will
replace this when access is granted.

Outputs: models/dikhow_v1_meta.json (+ dikhow_v1.pkl if ML ships),
data/history/model_v1_metrics.json, docs/MODEL-CARD-dikhow-v1.md
(+ served copy in public/).
"""

import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PUBLIC_DIR, REPO_ROOT, load_json, save_json, utc_now_iso  # noqa: E402

from baselines import fit_b3_slopes, b3_lag_rule  # noqa: E402
from lag_analysis import load_hourly_rain  # noqa: E402
from mdata import (HIST_DIR, build_daily_frame, feature_frame,  # noqa: E402
                   feature_cols, leakage_test, monsoon_mask, season_metrics,
                   train_q90)

MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR = REPO_ROOT / "docs"
YEARS = list(range(2015, 2026))
FOLDS_V1 = [(list(range(2015, 2024)), 2024), (list(range(2015, 2025)), 2025)]
H = 1                      # primary + only gated horizon (h=2 reported for shipped)
RAIN_FC_IDS = ["zunheboto", "satakha", "aghunato", "mokokchung",
               "longkhim", "changtongya", "naginimora", "mon"]

CAVEAT = ("Trained and evaluated against GloFAS v4 reanalysis - a MODELLED "
          "product, not observed river data. Observed CWC gauge data will "
          "replace this when access is granted. Predictions are modelled "
          "discharge, never 'river levels'.")

HGB_PARAMS = dict(max_iter=400, learning_rate=0.05, early_stopping=False,
                  random_state=0)
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=4,
                  subsample=0.9, random_state=0, n_jobs=2, verbosity=0)


def load_fc_daily() -> pd.DataFrame:
    """Catchment-mean issued-forecast daily totals, indexed by TARGET day:
    fc1_target(d) = day-d rain as forecast 1 day ahead; fc2_target(d) = as
    forecast 2 days ahead."""
    frames1, frames2 = [], []
    for pid in RAIN_FC_IDS:
        parts = []
        d = HIST_DIR / "rainfall_fc" / pid
        for pq in sorted(d.glob("*.parquet")):
            parts.append(pd.read_parquet(pq))
        if not parts:
            continue
        df = pd.concat(parts)
        idx = pd.to_datetime(df["time"])
        s1 = pd.Series(df["fc_prev1_mm"].to_numpy(), index=idx)
        s2 = pd.Series(df["fc_prev2_mm"].to_numpy(), index=idx)
        frames1.append(s1.groupby(s1.index.date).sum())
        frames2.append(s2.groupby(s2.index.date).sum())
    f1 = pd.concat(frames1, axis=1).mean(axis=1)
    f2 = pd.concat(frames2, axis=1).mean(axis=1)
    out = pd.DataFrame({"fc1_target": f1, "fc2_target": f2})
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def actual_next24(years) -> pd.Series:
    """ACTUAL day-d catchment rain total, indexed by day d (for the
    perfect-prog feature and the upper-bound evaluation)."""
    rain_h = load_hourly_rain(years)
    tot = rain_h.groupby(rain_h.index.date).sum()
    tot.index = pd.to_datetime(tot.index)
    return tot


def add_rainnext_feature(daily: pd.DataFrame, source: pd.Series) -> pd.DataFrame:
    """rain_next24(d) = the FEATURE for predicting day d+1: either actual
    day-(d+1) rain (perfect-prog training) or the issued forecast for
    day d+1 (operational evaluation). Stored at prediction day d."""
    out = daily.copy()
    out["rain_next24"] = source.reindex(out.index).shift(-1)
    return out


def fit_ml(feats: pd.DataFrame, train_years, model_kind: str):
    cols = feature_cols(feats)
    tr = feats[feats.index.year.isin(train_years)].dropna(subset=cols + [f"y_h{H}"])
    model = (HistGradientBoostingRegressor(**HGB_PARAMS) if model_kind == "hgb"
             else XGBRegressor(**XGB_PARAMS))
    model.fit(tr[cols], tr[f"y_h{H}"])
    return model, cols


def pred_for_days(model, feats, cols) -> pd.Series:
    ok = feats.dropna(subset=cols)
    return pd.Series(model.predict(ok[cols]), index=ok.index + pd.Timedelta(days=H))


def fit_b4_slope(daily_pp: pd.DataFrame, train_years) -> float:
    """dq(d+1) vs the rain_next24 feature (perfect-prog in training)."""
    tr = daily_pp[daily_pp.index.year.isin(train_years) & monsoon_mask(daily_pp.index)]
    x = tr["rain_next24"]
    y = tr["q"].shift(-1) - tr["q"]
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    denom = float((df["x"] ** 2).sum())
    return float((df["x"] * df["y"]).sum() / denom) if denom else 0.0


def b4_predict(daily_eval: pd.DataFrame, slope: float) -> pd.Series:
    """Prediction FOR day d made at d-1: q(d-1) + slope * feature(d-1)."""
    return daily_eval["q"].shift(1) + slope * daily_eval["rain_next24"].shift(1)


def main() -> int:
    now = utc_now_iso()
    base = build_daily_frame(YEARS)
    act24 = actual_next24(YEARS)
    fc = load_fc_daily()

    # perfect-prog frame (training) and operational frame (evaluation)
    daily_pp = add_rainnext_feature(base, act24)
    daily_op = add_rainnext_feature(base, fc["fc1_target"])
    print("Leakage test on the v1 frame (corrupt-the-future)...")
    leakage_test(daily_pp.drop(columns=["rain_next24"]))
    print("  base features PASS; rain_next24 issuance timing rests on the "
          "Previous Runs API contract (documented)")

    feats_pp = feature_frame(daily_pp)
    feats_pp["rain_next24"] = daily_pp["rain_next24"]
    lag = load_json(HIST_DIR / "lag_summary.json")
    b0_meta = load_json(MODELS_DIR / "dikhow_v0_meta.json")

    results = {"generated_at": now, "target_caveat": CAVEAT,
               "design": "perfect-prog training / real-issued-forecast validation",
               "folds": []}
    residuals = {"hgb": [], "xgb": [], "b4": []}
    wins = {"hgb": 0, "xgb": 0, "b4_vs_b3": 0}
    inner_choice = None

    for train_years, val_year in FOLDS_V1:
        thr = train_q90(base, train_years)
        val_mask = (base.index.year == val_year) & monsoon_mask(base.index)
        actual = base["q"][val_mask]
        pers = base["q"].shift(H)[val_mask]

        # --- reference: B3 (v0's rule, no fc) ---
        slopes3 = fit_b3_slopes(base, train_years, lag["median_event_lag_days"])
        b3 = b3_lag_rule(base, slopes3, H)[val_mask]
        m_b3 = season_metrics(actual, b3, pers, thr, H)

        # --- B4: simple rule + fc feature ---
        slope4 = fit_b4_slope(daily_pp, train_years)
        b4 = b4_predict(daily_op, slope4)[val_mask]
        m_b4 = season_metrics(actual, b4, pers, thr, H)
        residuals["b4"].extend([round(float(r), 1) for r in (actual - b4).dropna()])

        # --- ML candidates: train perfect-prog, evaluate operational ---
        fold_ml = {}
        # inner validation: last training monsoon, perfect-prog (no fc there)
        inner_year = train_years[-1]
        inner_train = [y for y in train_years if y != inner_year]
        inner_mask = (base.index.year == inner_year) & monsoon_mask(base.index)
        inner_scores = {}
        for kind in ("hgb", "xgb"):
            im, icols = fit_ml(feats_pp, inner_train, kind)
            ip = pred_for_days(im, feats_pp, icols).reindex(base.index)[inner_mask]
            inner_scores[kind] = season_metrics(base["q"][inner_mask], ip,
                                                base["q"].shift(H)[inner_mask],
                                                thr, H)["mae"]
        chosen = min(inner_scores, key=inner_scores.get)
        if inner_choice is None:
            inner_choice = chosen   # first fold fixes the gated candidate

        feats_op = feats_pp.copy()
        feats_op["rain_next24"] = daily_op["rain_next24"]
        for kind in ("hgb", "xgb"):
            model, cols = fit_ml(feats_pp, train_years, kind)
            pred = pred_for_days(model, feats_op, cols).reindex(base.index)[val_mask]
            m = season_metrics(actual, pred, pers, thr, H)
            fold_ml[kind] = m
            residuals[kind].extend([round(float(r), 1) for r in (actual - pred).dropna()])
            if m["mae"] < m_b4["mae"] and m["mae"] < m_b3["mae"]:
                wins[kind] += 1
            # perfect-prog upper bound (analysis only)
            pred_pp = pred_for_days(model, feats_pp, cols).reindex(base.index)[val_mask]
            fold_ml[f"{kind}_perfectprog"] = season_metrics(actual, pred_pp, pers, thr, H)
        if m_b4["mae"] < m_b3["mae"]:
            wins["b4_vs_b3"] += 1

        results["folds"].append({
            "train_years": f"{train_years[0]}-{train_years[-1]}",
            "validation_season": f"monsoon {val_year}",
            "threshold_q90_train_m3s": round(thr, 1),
            "b4_slope": round(slope4, 4),
            "inner_selection": {"year": inner_year, "scores": inner_scores,
                                "chosen": chosen},
            "metrics_h1": {"B3_no_fc": m_b3, "B4_fc_rule": m_b4, **fold_ml},
        })
        print(f"  monsoon {val_year}: MAE B3={m_b3['mae']} B4={m_b4['mae']} "
              f"HGB={fold_ml['hgb']['mae']} XGB={fold_ml['xgb']['mae']} "
              f"(perfect-prog {fold_ml['hgb_perfectprog']['mae']}/"
              f"{fold_ml['xgb_perfectprog']['mae']}) | B4 POD "
              f"{m_b4['pod_days']} FAR {m_b4['far_days']} lead {m_b4['mean_lead_days']}")

    # ---- pre-registered ship decision ----
    gated = inner_choice
    if wins[gated] == len(FOLDS_V1):
        shipped, ship_txt = f"ml_{gated}", (
            f"v1-{gated.upper()} beats B4 and B3 in BOTH covered seasons -> SHIP v1-ML")
    elif wins["b4_vs_b3"] == len(FOLDS_V1):
        shipped, ship_txt = "B4_fc_rule", (
            f"gated ML ({gated}) won {wins[gated]}/2; B4 beats B3 in both seasons "
            f"-> SHIP B4 (lag rule + forecast rain) as model v1")
    else:
        shipped, ship_txt = "keep_v0", (
            f"gated ML ({gated}) won {wins[gated]}/2 and B4 beat B3 in "
            f"{wins['b4_vs_b3']}/2 -> v1 NOT earned; model v0 (B3) stays shipped")
    results["ship_decision"] = {"gated_candidate": gated, "wins": wins,
                                "shipped": shipped, "statement": ship_txt}
    print(f"  DECISION: {ship_txt}")

    # ---- final artifact ----
    MODELS_DIR.mkdir(exist_ok=True)
    thr_final = train_q90(base, YEARS)
    meta = {"generated_at": now, "target_caveat": CAVEAT,
            "version": "v1" if shipped != "keep_v0" else "v0-retained",
            "shipped": shipped, "decision_statement": ship_txt,
            "design": results["design"],
            "threshold_q90_monsoon_2015_2025_m3s": round(thr_final, 1),
            "fc_coverage": "issued forecasts Jun 2024+ (Previous Runs API)"}
    if shipped == "B4_fc_rule":
        meta["b4_slope_full_period"] = round(fit_b4_slope(daily_pp, YEARS), 4)
        meta["validation_residuals_m3s"] = {"1": residuals["b4"]}
    elif shipped.startswith("ml_"):
        model, cols = fit_ml(feats_pp, YEARS, gated)
        joblib.dump({"model": model, "features": cols}, MODELS_DIR / "dikhow_v1.pkl")
        meta["model_file"] = "models/dikhow_v1.pkl"
        meta["validation_residuals_m3s"] = {"1": residuals[gated]}
    save_json(MODELS_DIR / "dikhow_v1_meta.json", meta)
    save_json(HIST_DIR / "model_v1_metrics.json", results)
    write_card(results, meta)
    print(f"OK models/dikhow_v1_meta.json + docs/MODEL-CARD-dikhow-v1.md "
          f"(shipped: {shipped})")
    return 0


def write_card(results: dict, meta: dict) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    L = []
    A = L.append
    A("# MODEL CARD - dikhow_v1 (forecast-rain upgrade attempt)")
    A("")
    A(f"> **{CAVEAT}**")
    A("")
    A(f"Generated {results['generated_at']} by `backend/model/train_v1.py`.")
    A("")
    A("## Ship decision (rule pre-registered before training - see the")
    A("Model v1 kickoff commit)")
    A("")
    A(f"**{results['ship_decision']['statement']}**")
    A("")
    A("## Design")
    A("")
    A("- New feature: catchment rain for the TARGET day as forecast 1 day")
    A("  ahead. Training uses the ACTUAL next-day rain (perfect-prog);")
    A("  validation feeds the REAL forecasts issued at the time (Previous")
    A("  Runs archive, non-null from Jun 2024 only) - so forecast error is")
    A("  in the held-out numbers.")
    A("- Only two operationally honest held-out monsoons exist (2024, 2025).")
    A("  That is a small sample and this card says so.")
    A("- Candidates: B4 = v0's lag rule + the fc feature (slope fitted on")
    A("  training only); v1-HGB and v1-XGB (gated candidate picked by inner")
    A("  validation on the last training monsoon, no held-out shopping).")
    A("- Perfect-prog rows below are the ANALYSIS-ONLY upper bound (what a")
    A("  perfect rain forecast would buy); they are never shippable.")
    A("")
    A("## Held-out monsoon metrics, horizon 1 day")
    A("")
    A("| season | B3 (v0, no fc) | B4 (fc rule) | HGB | XGB | HGB perfect-prog | XGB perfect-prog |")
    A("|---|---|---|---|---|---|---|")
    for f in results["folds"]:
        m = f["metrics_h1"]
        A(f"| {f['validation_season']} | {m['B3_no_fc']['mae']} | {m['B4_fc_rule']['mae']} "
          f"| {m['hgb']['mae']} | {m['xgb']['mae']} "
          f"| {m['hgb_perfectprog']['mae']} | {m['xgb_perfectprog']['mae']} |")
    A("")
    A("Event stats (B4 and gated ML), per season:")
    A("")
    A("| season | candidate | POD | FAR | mean lead (d) |")
    A("|---|---|---|---|---|")
    gated = results["ship_decision"]["gated_candidate"]
    for f in results["folds"]:
        m = f["metrics_h1"]
        for name, key in (("B4", "B4_fc_rule"), (gated.upper(), gated)):
            A(f"| {f['validation_season']} | {name} | {m[key]['pod_days']} "
              f"| {m[key]['far_days']} | {m[key]['mean_lead_days']} |")
    A("")
    A("No blended accuracy number is reported, deliberately.")
    A("")
    A("## Failure modes (v0's list still applies, plus)")
    A("")
    A("- v1 inherits the WEATHER model's misses: if the issued forecast has")
    A("  no rain, v1 is as blind as v0 (the perfect-prog gap above measures")
    A("  exactly this).")
    A("- Two held-out seasons is a thin validation base; treat all v1 skill")
    A("  claims as provisional until more monsoons accumulate.")
    A("- Dam releases, cloudburst smoothing, GloFAS-vs-GloFAS error")
    A("  correlation, daily-step blindness: unchanged from the v0 card.")
    A("")
    card = "\n".join(L) + "\n"
    (DOCS_DIR / "MODEL-CARD-dikhow-v1.md").write_text(card, encoding="utf-8")
    (PUBLIC_DIR / "MODEL-CARD-dikhow-v1.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
