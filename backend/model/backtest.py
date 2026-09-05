"""PHASE 1 (AI-ready platform) - the BACKTEST CLI + model registry.

    python backend/model/backtest.py --basin dikhow --candidate B3,hgb \
        --seasons 2022-2025 --horizon 1 [--register] [--verify-v0]

One code path for every candidate, one metrics record per (basin, season,
horizon, candidate) appended to models/registry.json with the data
manifest hash and git revision - so every number anyone quotes can be
traced to the exact data and code that produced it.

Candidates: persistence, climatology, B3 (lag rule), hgb (gradient
boosting, v0 features). Walk-forward: train = 2015..season-1, validate =
that season's monsoon (Jun-Sep). Never a random split.

--verify-v0 asserts that the B3 and hgb numbers reproduce the committed
v0 artifacts (baseline_metrics.json / model_metrics.json) to the decimal.

Target caveat: GloFAS v4 reanalysis - a MODELLED product, not observed
river data; observed CWC gauge data will replace it when access is granted.
"""

import argparse
import hashlib
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import REPO_ROOT, load_json, save_json, utc_now_iso  # noqa: E402

from baselines import b1_persistence, b2_climatology, b3_lag_rule, fit_b3_slopes  # noqa: E402
from mdata import (HIST_DIR, build_daily_frame, feature_frame, monsoon_mask,  # noqa: E402
                   season_metrics, train_q90)
from train import fit_fold, predict_for_days  # noqa: E402

REGISTRY = REPO_ROOT / "models" / "registry.json"
CANDIDATES = ["persistence", "climatology", "B3", "hgb"]


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def manifest_hash() -> str | None:
    m = HIST_DIR / "MANIFEST.json"
    if not m.exists():
        return None
    return hashlib.sha256(m.read_bytes()).hexdigest()[:16]


def seasons_arg(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def run(basin: str, candidates: list[str], seasons: list[int], horizon: int) -> list[dict]:
    lag_file = "lag_summary.json" if basin == "dikhow" else f"lag_summary_{basin}.json"
    if not (HIST_DIR / lag_file).exists():
        raise SystemExit(f"basin {basin!r}: run fetch_history.py --basin {basin} and "
                         f"lag_analysis.py --basin {basin} first")
    years = list(range(2015, max(seasons) + 1))
    daily = build_daily_frame(years, basin)
    feats = feature_frame(daily) if "hgb" in candidates else None
    lag = load_json(HIST_DIR / lag_file)
    records = []
    for season in seasons:
        train_years = [y for y in years if y < season]
        thr = train_q90(daily, train_years)
        val_mask = (daily.index.year == season) & monsoon_mask(daily.index)
        actual = daily["q"][val_mask]
        pers = b1_persistence(daily, horizon)[val_mask]
        for cand in candidates:
            if cand == "persistence":
                pred = pers
            elif cand == "climatology":
                pred = b2_climatology(daily, train_years, horizon)[val_mask]
            elif cand == "B3":
                slopes = fit_b3_slopes(daily, train_years, lag["median_event_lag_days"])
                pred = b3_lag_rule(daily, slopes, horizon)[val_mask]
            elif cand == "hgb":
                model, cols = fit_fold(feats, train_years, horizon)
                pred = predict_for_days(model, feats, cols, horizon).reindex(daily.index)[val_mask]
            else:
                raise SystemExit(f"unknown candidate {cand}")
            m = season_metrics(actual, pred, pers, thr, horizon)
            records.append({"basin": basin, "candidate": cand, "season": f"monsoon {season}",
                            "train_years": f"{train_years[0]}-{train_years[-1]}",
                            "horizon_days": horizon, "metrics": m})
    return records


def verify_v0(records: list[dict]) -> None:
    base = load_json(HIST_DIR / "baseline_metrics.json")
    model = load_json(HIST_DIR / "model_metrics.json")
    by_season_b = {f["validation_season"]: f for f in base["folds"]}
    by_season_m = {f["validation_season"]: f for f in model["folds"]}
    n = 0
    for r in records:
        h = f"h{r['horizon_days']}"
        if r["candidate"] == "B3" and r["season"] in by_season_b:
            ref = by_season_b[r["season"]]["baselines"][h]["B3_lag_rule"]
        elif r["candidate"] == "hgb" and r["season"] in by_season_m:
            ref = by_season_m[r["season"]]["horizons"][h]
        elif r["candidate"] == "persistence" and r["season"] in by_season_b:
            ref = by_season_b[r["season"]]["baselines"][h]["B1_persistence"]
        elif r["candidate"] == "climatology" and r["season"] in by_season_b:
            ref = by_season_b[r["season"]]["baselines"][h]["B2_climatology"]
        else:
            continue
        for k in ("mae", "rmse", "pod_days", "far_days", "mean_lead_days"):
            assert r["metrics"][k] == ref[k], \
                f"MISMATCH {r['candidate']} {r['season']} {k}: backtest {r['metrics'][k]} vs v0 card {ref[k]}"
        n += 1
    print(f"  verify-v0: {n} records reproduce the committed v0 artifacts to the decimal")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", default="dikhow")
    ap.add_argument("--candidate", default="persistence,climatology,B3,hgb")
    ap.add_argument("--seasons", default="2022-2025")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--register", action="store_true", help="append to models/registry.json")
    ap.add_argument("--verify-v0", action="store_true")
    a = ap.parse_args()
    cands = [c.strip() for c in a.candidate.split(",")]
    records = run(a.basin, cands, seasons_arg(a.seasons), a.horizon)

    print(f"backtest basin={a.basin} horizon={a.horizon} d - held-out monsoon MAE (m3/s):")
    for r in records:
        m = r["metrics"]
        print(f"  {r['season']:13s} {r['candidate']:12s} MAE {m['mae']:6.1f}  RMSE {m['rmse']:6.1f}  "
              f"skill {m['skill_vs_persistence']}  POD {m['pod_days']}  FAR {m['far_days']}  "
              f"lead {m['mean_lead_days']} d")
    if a.verify_v0:
        verify_v0(records)
    if a.register:
        reg = load_json(REGISTRY) if REGISTRY.exists() else {"records": []}
        stamp = {"run_at": utc_now_iso(), "git_rev": git_rev(), "manifest_hash": manifest_hash(),
                 "target_caveat": "GloFAS v4 reanalysis (modelled product, not observed river data)"}
        for r in records:
            reg["records"].append({**stamp, **r})
        save_json(REGISTRY, reg)
        print(f"  registered {len(records)} records -> models/registry.json "
              f"({len(reg['records'])} total, git {stamp['git_rev']}, manifest {stamp['manifest_hash']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
