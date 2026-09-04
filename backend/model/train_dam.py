"""DAM-FEATURE RETRAIN - pre-registered before results were seen.

Question: does knowing each dam's reservoir fill let a model beat the
simple baseline on the DAM-CONTROLLED basins, where rain features could
not? Candidates (they currently ship a non-ML baseline, and we hold a
clean multi-year dam-fill series for their reservoir):

    kopili    <- Kopili + Khandong reservoirs   (currently B1 persistence)
    dhansiri  <- Doyang reservoir               (currently B1 persistence)
    ranganadi <- Ranganadi + Pare reservoirs    (currently trained model)

Dam feature: norm_level (per-dam level normalised over its own 7-year
2nd/98th percentile, from reservoirs_clean.parquet), plus its 1-day lag
and day-change, joined to the basin's feature frame by date. NERLDC
publishes the previous day's report, so at prediction time the dam level
is known with ~1-day latency - the lagged columns respect that.

SHIP RULE (fixed here, same as every basin): the dam-augmented
HistGradientBoosting model (same params as the other basins) replaces
the current shipped model ONLY if it beats that basin's current baseline
(persistence for kopili/dhansiri) on monsoon MAE at h=1 in >= 3 of 4
held-out monsoons (2022-2025, walk-forward). We ALSO report the plain
(no-dam) model, so any gain is attributable to the dam feature and not
just to switching to ML.

HONEST CAVEAT stated up front: the target is GloFAS v4 reanalysis, a
hydrological model that may not represent dam operations well - so a dam
feature could fail to help predict this target even if it would help
predict real gauged flow. A negative here is therefore not the final
word; the real test awaits observed CWC levels. Either outcome is
documented, nothing is tuned after seeing results.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "backend" / "model"))
from mdata import build_daily_frame, feature_frame, feature_cols, monsoon_mask  # noqa: E402

CLEAN = BASE / "data" / "history" / "nerldc" / "reservoirs_clean.parquet"
OUT = BASE / "data" / "history" / "dam_retrain.json"
HGB = {"max_iter": 400, "learning_rate": 0.05, "random_state": 0}
FOLDS = [(list(range(2015, fy)), fy) for fy in (2022, 2023, 2024, 2025)]
DAMS = {"kopili": ["Kopili", "Khandong"],
        "ranganadi": ["Ranganadi", "Pare"],
        "dhansiri": ["Doyang"]}


def dam_features(basin: str) -> pd.DataFrame:
    d = pd.read_parquet(CLEAN)
    d = d[d["reservoir"].isin(DAMS[basin])]
    if d.empty:
        return pd.DataFrame()
    wide = d.pivot_table(index="date", columns="reservoir", values="norm_level")
    wide.index = pd.to_datetime(wide.index)
    out = pd.DataFrame(index=wide.index)
    for res in DAMS[basin]:
        if res in wide:
            out[f"dam_{res}"] = wide[res]
            out[f"dam_{res}_lag1"] = wide[res].shift(1)
            out[f"dam_{res}_chg"] = wide[res].diff()
    return out


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def run(basin: str) -> dict:
    years = list(range(2015, 2026))
    daily = build_daily_frame(years, basin)
    feats = feature_frame(daily)
    dam = dam_features(basin)
    if dam.empty:
        return {"basin": basin, "skipped": "no dam series"}
    feats = feats.join(dam)
    base_cols = feature_cols(feats)
    dam_cols = [c for c in feats.columns if c.startswith("dam_")]
    plain_cols = [c for c in base_cols if c not in dam_cols]

    y = feats["y_h1"]
    monsoon_train = "--monsoon-train" in sys.argv
    rows = []
    wins_vs_pers = wins_vs_plain = 0
    for train_years, fy in FOLDS:
        tr = feats.index.year.isin(train_years) & y.notna()
        if monsoon_train:
            tr = tr & monsoon_mask(feats.index)   # train on monsoon days only
        te = (feats.index.year == fy) & monsoon_mask(feats.index) & y.notna()
        if te.sum() < 30 or feats.loc[te, dam_cols].notna().mean().mean() < 0.5:
            rows.append({"fold": fy, "skipped": "thin dam coverage"}); continue
        pers = mae(y[te], feats.loc[te, "q"])                       # persistence
        m_plain = HistGradientBoostingRegressor(**HGB).fit(feats.loc[tr, plain_cols], y[tr])
        m_dam = HistGradientBoostingRegressor(**HGB).fit(feats.loc[tr, base_cols], y[tr])
        e_plain = mae(y[te], m_plain.predict(feats.loc[te, plain_cols]))
        e_dam = mae(y[te], m_dam.predict(feats.loc[te, base_cols]))
        wins_vs_pers += e_dam < pers
        wins_vs_plain += e_dam < e_plain
        rows.append({"fold": fy, "persistence": round(pers, 1),
                     "plain_ml": round(e_plain, 1), "dam_ml": round(e_dam, 1)})
    judged = [r for r in rows if "dam_ml" in r]
    ship = len(judged) >= 3 and wins_vs_pers >= 3
    verdict = (f"dam model beats persistence in {wins_vs_pers}/{len(judged)} monsoons "
               f"(and the no-dam model in {wins_vs_plain}/{len(judged)}) -> "
               + ("SHIP dam model" if ship else "KEEP current baseline"))
    return {"basin": basin, "dam_reservoirs": DAMS[basin], "folds": rows,
            "wins_vs_persistence": wins_vs_pers, "wins_vs_plain_ml": wins_vs_plain,
            "ship": ship, "verdict": verdict,
            "caveat": "target is GloFAS reanalysis (modelled), may not reflect dam ops"}


def main() -> int:
    if not CLEAN.exists():
        print("run clean_reservoirs.py first (need reservoirs_clean.parquet)")
        return 0
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or list(DAMS)
    if "--monsoon-train" in sys.argv:
        print("(training on MONSOON days only - testing the user's variant)\n")
    results = []
    for basin in which:
        if basin not in DAMS:
            continue
        r = run(basin)
        results.append(r)
        print(f"\n=== {basin} ===")
        for f in r.get("folds", []):
            print("  ", f)
        print("  DECISION:", r.get("verdict", r.get("skipped")))
    OUT.write_text(json.dumps({"results": results,
                               "rule": "dam ML ships only if it beats persistence in >=3/4 held-out monsoons"},
                              indent=1), encoding="utf-8")
    print(f"\nwritten -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
