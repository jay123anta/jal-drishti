"""MODEL v0 (generic per basin since Phase 1c) - train + walk-forward-validate.

    python backend/model/train.py                  # dikhow (default)
    python backend/model/train.py --basin kopili

Model: sklearn HistGradientBoostingRegressor, one regressor per horizon
(1 d, 2 d). Features: recent discharge and its changes, rolling catchment
rain sums (6/12/24/48 h + yesterday's 24 h), antecedent wetness (14/30 d),
day-of-year, upstream GloFAS cells where resolved - all leakage-tested
(mdata.leakage_test corrupts the future and asserts features are unchanged).

Validation: walk-forward by monsoon season ONLY (train 2015-2021 -> monsoon
2022; extend -> 2023; -> 2024; -> 2025). Metrics per fold, never on
training years, never blended.

SHIP RULE (pre-registered in internal records "Phase 1c" BEFORE any new basin was
trained): best baseline = the lower MEAN held-out h=1 MAE of B1 persistence
/ B3 lag rule (tie -> B3). The trained model ships as model v0 only if it
beats the best baseline in >= 3 of 4 held-out seasons; otherwise the best
baseline ships AS model v0 and the card says so. (For the Dikhow this
reproduces the original M4 decision.)

Target caveat everywhere: trained and evaluated against GloFAS v4
reanalysis - a MODELLED product, not observed river data; observed CWC
gauge data will replace this when access is granted.

Outputs (names per basin from basins.model_names): models/<meta>,
models/<pkl> if ML ships, data/history/<metrics>, docs/<card> (+ public/).
"""

import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import PUBLIC_DIR, REPO_ROOT, load_json, save_json, utc_now_iso  # noqa: E402

from baselines import b3_lag_rule, fit_b3_slopes  # noqa: E402
from basins import BASINS, model_names  # noqa: E402
from mdata import (FOLDS, HIST_DIR, HORIZONS, PRIMARY_HORIZON,  # noqa: E402
                   build_daily_frame, feature_cols, feature_frame,
                   leakage_test, monsoon_mask, season_metrics, train_q90)

MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR = REPO_ROOT / "docs"
YEARS = list(range(2015, 2026))

CAVEAT = ("Trained and evaluated against GloFAS v4 reanalysis - a MODELLED "
          "product, not observed river data. Observed CWC gauge data will "
          "replace this when access is granted. Predictions are modelled "
          "discharge, never 'river levels'.")

HGB_PARAMS = dict(max_iter=400, learning_rate=0.05, early_stopping=False,
                  random_state=0)


def fit_fold(feats: pd.DataFrame, train_years, h: int):
    cols = feature_cols(feats)
    tr = feats[feats.index.year.isin(train_years)].dropna(subset=cols + [f"y_h{h}"])
    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(tr[cols], tr[f"y_h{h}"])
    return model, cols


def predict_for_days(model, feats: pd.DataFrame, cols, h: int) -> pd.Series:
    """Model prediction FOR day d (made at day d-h)."""
    ok = feats.dropna(subset=cols)
    yhat = model.predict(ok[cols])
    return pd.Series(yhat, index=ok.index + pd.Timedelta(days=h))


def main() -> int:
    args = sys.argv[1:]
    basin = args[args.index("--basin") + 1] if "--basin" in args else "dikhow"
    if basin not in BASINS:
        raise SystemExit(f"unknown basin {basin!r}; known: {list(BASINS)}")
    nm = model_names(basin)
    now = utc_now_iso()

    daily = build_daily_frame(YEARS, basin)
    print(f"[{basin}] leakage test (corrupt-the-future)...")
    leakage_test(daily)
    print("  leakage test PASSED - features use only data <= prediction day")
    feats = feature_frame(daily)
    baselines = load_json(HIST_DIR / nm["baselines"])
    lag = load_json(HIST_DIR / nm["lag"])

    # ---- best baseline (pre-registered rule) ----
    def mean_mae(key):
        return float(np.mean([f["baselines"]["h1"][key]["mae"] for f in baselines["folds"]]))
    b1_mean, b3_mean = mean_mae("B1_persistence"), mean_mae("B3_lag_rule")
    best_key = "B3_lag_rule" if b3_mean <= b1_mean else "B1_persistence"

    metrics = {"generated_at": now, "basin": basin, "target_caveat": CAVEAT,
               "model": "sklearn HistGradientBoostingRegressor", "params": HGB_PARAMS,
               "features": feature_cols(feats),
               "best_baseline": {"key": best_key, "mean_mae_B1": round(b1_mean, 1),
                                 "mean_mae_B3": round(b3_mean, 1)},
               "folds": []}
    residuals = {"trained": {h: [] for h in HORIZONS},
                 "B3_lag_rule": {h: [] for h in HORIZONS},
                 "B1_persistence": {h: [] for h in HORIZONS}}
    wins = 0

    for fold_i, (train_years, val_year) in enumerate(FOLDS):
        thr = train_q90(daily, train_years)
        slopes = fit_b3_slopes(daily, train_years, lag["median_event_lag_days"])
        val_mask = (daily.index.year == val_year) & monsoon_mask(daily.index)
        actual = daily["q"][val_mask]
        fold = {"train_years": f"{train_years[0]}-{train_years[-1]}",
                "validation_season": f"monsoon {val_year}",
                "threshold_q90_train_m3s": round(thr, 1), "horizons": {}}
        for h in HORIZONS:
            model, cols = fit_fold(feats, train_years, h)
            pred = predict_for_days(model, feats, cols, h).reindex(daily.index)[val_mask]
            pers = daily["q"].shift(h)[val_mask]
            m = season_metrics(actual, pred, pers, thr, h)
            fold["horizons"][f"h{h}"] = m
            residuals["trained"][h].extend([round(float(r), 1) for r in (actual - pred).dropna()])
            residuals["B3_lag_rule"][h].extend(
                [round(float(r), 1) for r in (actual - b3_lag_rule(daily, slopes, h)[val_mask]).dropna()])
            residuals["B1_persistence"][h].extend(
                [round(float(r), 1) for r in (actual - pers).dropna()])
            if h == PRIMARY_HORIZON:
                best = baselines["folds"][fold_i]["baselines"]["h1"][best_key]
                if m["mae"] < best["mae"]:
                    wins += 1
        metrics["folds"].append(fold)
        h1 = fold["horizons"]["h1"]
        bb = baselines["folds"][fold_i]["baselines"]["h1"][best_key]
        print(f"  {fold['validation_season']}: h1 MAE model={h1['mae']} vs {best_key}="
              f"{bb['mae']} | POD {h1['pod_days']} FAR {h1['far_days']} lead {h1['mean_lead_days']} d")

    ship_trained = wins >= 3
    shipped = "trained_hgb" if ship_trained else best_key
    decision = (f"trained model beats the best baseline ({best_key}, mean MAE "
                f"{min(b1_mean, b3_mean):.1f}) on monsoon MAE at h=1 in {wins}/4 held-out "
                f"seasons -> " + ("SHIP TRAINED MODEL" if ship_trained else
                                  f"SHIP {best_key} as model v0 (an honest simple model "
                                  f"beats a dressed-up failure)"))
    metrics["ship_decision"] = {"wins_h1_vs_best_baseline": wins, "of_folds": len(FOLDS),
                                "best_baseline": best_key, "shipped": shipped,
                                "statement": decision}
    print(f"  DECISION: {decision}")

    # ---- final shipped artifact (fitted on ALL 2015-2025) ----
    MODELS_DIR.mkdir(exist_ok=True)
    thr_final = train_q90(daily, YEARS)
    slopes_final = fit_b3_slopes(daily, YEARS, lag["median_event_lag_days"])
    res_key = "trained" if ship_trained else best_key
    meta = {
        "generated_at": now, "basin": basin, "target": BASINS[basin]["target"],
        "target_caveat": CAVEAT,
        "shipped": shipped, "decision_statement": decision,
        "best_baseline": best_key,
        "horizons_days": HORIZONS, "primary_horizon_days": PRIMARY_HORIZON,
        "features": feature_cols(feats),
        "threshold_q90_monsoon_2015_2025_m3s": round(thr_final, 1),
        "b3_slopes_full_period": {str(h): round(s, 4) for h, s in slopes_final.items()},
        "b3_lag_days": lag["median_event_lag_days"],
        "validation_residuals_m3s": {str(h): residuals[res_key][h] for h in HORIZONS},
        "residuals_note": ("held-out walk-forward residuals (actual - predicted) OF THE "
                           "SHIPPED MODEL, pooled across the 4 validation monsoons; used "
                           "for exceedance probability"),
    }
    if ship_trained:
        final_models = {}
        for h in HORIZONS:
            model, cols = fit_fold(feats, YEARS, h)
            final_models[f"h{h}"] = model
        joblib.dump({"models": final_models, "features": feature_cols(feats)},
                    MODELS_DIR / nm["pkl"])
        meta["model_file"] = f"models/{nm['pkl']}"
    save_json(MODELS_DIR / nm["meta"], meta)
    save_json(HIST_DIR / nm["metrics"], metrics)
    write_model_card(basin, nm, metrics, baselines, meta)

    assert (MODELS_DIR / nm["meta"]).exists() and (DOCS_DIR / nm["card"]).exists()
    if ship_trained:
        assert (MODELS_DIR / nm["pkl"]).exists()
    print(f"OK models/{nm['meta']} + docs/{nm['card']} (shipped: {shipped})")
    return 0


def write_model_card(basin: str, nm: dict, metrics: dict, baselines: dict, meta: dict) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    cfg = BASINS[basin]
    L = []
    A = L.append
    A(f"# MODEL CARD - {basin}_v0 ({cfg['label']} discharge forecast)")
    A("")
    A(f"> **{CAVEAT}**")
    A("")
    A(f"Generated {metrics['generated_at']} by `backend/model/train.py --basin {basin}`.")
    A("")
    A("## What ships as model v0")
    A("")
    A(f"**{metrics['ship_decision']['statement']}**")
    A("")
    A(f"Rule (pre-registered before training, internal records 'Phase 1c'): best baseline = lower "
      f"mean held-out h=1 MAE of B1 persistence ({metrics['best_baseline']['mean_mae_B1']}) / "
      f"B3 lag rule ({metrics['best_baseline']['mean_mae_B3']}), tie -> B3; ML ships only if "
      f"it beats it in >= 3 of 4 seasons.")
    A("")
    A("## Data")
    A("")
    A(f"- Rainfall: hourly ERA5-family reanalysis, {len(cfg['rain_points'])} catchment anchor "
      f"cells (CATCHMENT.md), 2015-2025, Open-Meteo Archive API.")
    A(f"- Discharge: DAILY GloFAS v4 reanalysis at the target cell `{cfg['target']}` "
      f"plus resolved upstream cells: {', '.join(u['id'] for u in cfg['upstream'])} "
      f"(kept only where the probe found a plausible river cell).")
    A("- 2026 (Jan-Aug 5) fetched separately, test-only, never trained on.")
    A("- Gaps: see data/history/GAPS.md. Backfill state: BACKFILL-STATE.json.")
    A("")
    A("## Features (leakage-tested)")
    A("")
    A(", ".join(f"`{c}`" for c in metrics["features"]))
    A("")
    A("## Horizons")
    A("")
    A("1 day (primary) and 2 days. GloFAS is daily; 6/12 h horizons do not")
    A("exist against this target and are not reported.")
    A("")
    A("## Walk-forward validation (held-out monsoon seasons only)")
    A("")
    A("Threshold for event stats = 90th percentile of TRAINING-years monsoon")
    A("discharge (per fold, no leakage). POD/FAR are day-basis; lead is per")
    A("event onset (persistence scores 0 by construction; NEGATIVE lead means")
    A("the first correct alarm came only after the event had already begun).")
    A("")
    for h in HORIZONS:
        A(f"### Horizon {h} day{'s' if h > 1 else ''}")
        A("")
        A("| season | model MAE | B1 pers | B2 clim | B3 lag | model RMSE | model skill | POD | FAR | lead (d) |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for fi, fold in enumerate(metrics["folds"]):
            m = fold["horizons"][f"h{h}"]
            bl = baselines["folds"][fi]["baselines"][f"h{h}"]
            A(f"| {fold['validation_season']} | **{m['mae']}** "
              f"| {bl['B1_persistence']['mae']} | {bl['B2_climatology']['mae']} "
              f"| {bl['B3_lag_rule']['mae']} | {m['rmse']} "
              f"| {m['skill_vs_persistence']} | {m['pod_days']} "
              f"| {m['far_days']} | {m['mean_lead_days']} |")
        A("")
        A("Baseline event stats (B3) per season for comparison: " +
          "; ".join(f"{baselines['folds'][fi]['validation_season']}: "
                    f"POD {baselines['folds'][fi]['baselines'][f'h{h}']['B3_lag_rule']['pod_days']}, "
                    f"FAR {baselines['folds'][fi]['baselines'][f'h{h}']['B3_lag_rule']['far_days']}, "
                    f"lead {baselines['folds'][fi]['baselines'][f'h{h}']['B3_lag_rule']['mean_lead_days']} d"
                    for fi in range(len(metrics["folds"]))))
        A("")
    A("No aggregate/blended accuracy number is reported, deliberately: every")
    A("metric above is tied to one season, one basin, one horizon.")
    A("")
    A("## Operational conversion to colours")
    A("")
    A(f"Exceedance threshold: {meta['threshold_q90_monsoon_2015_2025_m3s']} m³/s")
    A("(90th percentile of 2015-2025 monsoon reanalysis). Exceedance")
    A("probability = share of pooled held-out residuals of the SHIPPED model")
    A("that would lift the point forecast over the threshold. Cutoffs")
    A("(documented, arbitrary): P >= 0.5 -> RED, P >= 0.2 -> YELLOW, else GREEN.")
    A("")
    A("## Failure modes and known-unsafe conditions")
    A("")
    A("- **Dam/barrage releases and channel changes are invisible** to")
    A("  rainfall-driven features; a release-driven surge will be missed"
      + (" - the Kopili is regulated (Umrongso), so this applies with force." if basin == "kopili" else "."))
    A("- Localized cloudbursts are heavily smoothed at reanalysis cell size;")
    A("  the model under-sees exactly the sharpest events (July 2026 class).")
    A("- The target is GloFAS's own routing of (partly) the same")
    A("  precipitation - errors correlate; real-river skill is UNKNOWN until")
    A("  CWC gauge data is available (Track T2 archive is accumulating).")
    A("- Daily steps: anything developing and peaking within one day is")
    A("  invisible between daily values.")
    A("- Trained on 2015-2025 only; regime changes are out of distribution.")
    A("- Monsoon-only validation: dry-season skill is unmeasured.")
    A("")
    card = "\n".join(L) + "\n"
    (DOCS_DIR / nm["card"]).write_text(card, encoding="utf-8")
    (PUBLIC_DIR / nm["card"]).write_text(card, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
