"""MODEL v0 (generic per basin since Phase 1c) - 2026 TRUE out-of-sample test.

    python backend/model/test_2026.py                  # dikhow
    python backend/model/test_2026.py --basin kopili

Runs the basin's SHIPPED model v0 (trained_hgb / B3_lag_rule /
B1_persistence, fitted through 2025 ONLY) over the 2026 partial history
partition (Jan 1 - Aug 5, test-only, never trained on) and reports
predicted vs actual around the July 2026 event EXACTLY as it comes out.

Definitions (same as the evaluation): prediction FOR day d at horizon h is
made at d-h from data <= d-h; threshold = the basin's q90 of 2015-2025
TRAINING monsoons (no 2026 leakage); alarm at t = prediction for t+h >=
threshold; lead(onset) = onset - first alarm time whose target day is in
the event.

Outputs (basins.model_names): public/<test> (chart data, actual OBSERVED,
predictions FORECAST + replay_test), public/<test_png>, and
data/history/<section> which analyze_replay.py appends to REPLAY-FINDINGS.md.
Pipeline-safe: skips (exit 0) if the model or history is missing.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import OBSERVED, PUBLIC_DIR, REPO_ROOT, load_json, save_json, utc_now_iso  # noqa: E402

from basins import BASINS, model_names  # noqa: E402
from mdata import HIST_DIR, HORIZONS, build_daily_frame, feature_frame  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
EVENT_DAY = pd.Timestamp("2026-07-19")
WINDOW = (pd.Timestamp("2026-06-15"), pd.Timestamp("2026-08-05"))
SRC_ACT = ("GloFAS v4 reanalysis via Open-Meteo Flood API (MODELLED product, "
           "not observed river data)")


def predictions(daily: pd.DataFrame, meta: dict, nm: dict) -> dict:
    """{h: Series of prediction FOR day d} for the shipped model."""
    shipped = meta["shipped"]
    if shipped == "trained_hgb":
        from train import predict_for_days
        bundle = __import__("joblib").load(MODELS_DIR / nm["pkl"])
        feats = feature_frame(daily)
        return {h: predict_for_days(bundle["models"][f"h{h}"], feats, bundle["features"], h)
                .reindex(daily.index) for h in HORIZONS}
    if shipped == "B3_lag_rule":
        s = meta["b3_slopes_full_period"]
        return {1: daily["q"].shift(1) + s["1"] * daily["rain_24h"].shift(1),
                2: daily["q"].shift(2) + s["2"] * daily["rain_48h"].shift(2)}
    return {h: daily["q"].shift(h) for h in HORIZONS}          # B1 persistence


def exceed_prob(pred: float, residuals: list, thr: float) -> float:
    return float(np.mean(pred + np.asarray(residuals) >= thr))


def main() -> int:
    args = sys.argv[1:]
    basin = args[args.index("--basin") + 1] if "--basin" in args else "dikhow"
    nm, cfg = model_names(basin), BASINS[basin]
    now = utc_now_iso()
    try:
        meta = load_json(MODELS_DIR / nm["meta"])
        assert (HIST_DIR / "discharge" / cfg["target"] / "2026.parquet").exists()
    except (FileNotFoundError, ValueError, AssertionError) as err:
        print(f"  [{basin}] 2026 test SKIPPED (degraded): {err or 'history missing'}")
        return 0
    thr = meta["threshold_q90_monsoon_2015_2025_m3s"]
    residuals = meta["validation_residuals_m3s"]
    src_pred = (f"JalDrishti model v0 for {basin} (shipped: {meta['shipped']}, fitted "
                f"2015-2025) - retrospective OUT-OF-SAMPLE prediction for the 2026 test; "
                f"target is GloFAS v4 reanalysis (modelled product, not observed river data)")

    daily = build_daily_frame([2026], basin)
    lo, hi = WINDOW
    preds_all = predictions(daily, meta, nm)
    actual = daily["q"].loc[lo:hi]
    preds = {h: preds_all[h].loc[lo:hi] for h in HORIZONS}
    probs = {h: preds[h].apply(lambda v: exceed_prob(v, residuals[str(h)], thr)
                               if pd.notna(v) else np.nan) for h in HORIZONS}
    alarms = {h: preds[h] >= thr for h in HORIZONS}

    # events = contiguous actual exceedance runs
    onsets, prev = [], False
    for d, ex in (actual >= thr).items():
        if ex and not prev:
            onsets.append(d)
        prev = ex
    leads = {}
    for o in onsets:
        run, d = [], o
        while d in actual.index and actual[d] >= thr:
            run.append(d)
            d += pd.Timedelta(days=1)
        at = [dd - pd.Timedelta(days=1) for dd in run if dd in alarms[1].index and alarms[1][dd]]
        leads[o] = (o - min(at)).days if at else None
    first_alarm_t = next((t - pd.Timedelta(days=1) for t, a in alarms[1].items() if a), None)

    def wrap(v, cls, src, **extra):
        return {"value": (None if pd.isna(v) else round(float(v), 1)), "class": cls,
                "source": src, "retrieved_at": now, **extra}

    doc = {
        "generated_at": now, "basin": basin, "test": "2026_out_of_sample",
        "target_caveat": meta["target_caveat"], "shipped_model": meta["shipped"],
        "decision_statement": meta["decision_statement"],
        "threshold_m3s": {"value": thr, "class": OBSERVED, "retrieved_at": now,
                          "source": "90th percentile of 2015-2025 TRAINING monsoon GloFAS "
                                    "reanalysis (no 2026 leakage)"},
        "dates": [d.strftime("%Y-%m-%d") for d in actual.index],
        "actual_m3s": [wrap(v, OBSERVED, SRC_ACT) for v in actual],
        "predicted_h1_m3s": [wrap(v, "FORECAST", src_pred, replay_test=True) for v in preds[1]],
        "predicted_h2_m3s": [wrap(v, "FORECAST", src_pred, replay_test=True) for v in preds[2]],
        "p_exceed_h1": [None if pd.isna(v) else round(float(v), 3) for v in probs[1]],
        "alarm_days_h1": [d.strftime("%Y-%m-%d") for d, a in alarms[1].items() if a],
        "actual_exceed_days": [d.strftime("%Y-%m-%d") for d, e in (actual >= thr).items() if e],
        "event_onsets": [{"onset": o.strftime("%Y-%m-%d"), "lead_days": leads[o]} for o in onsets],
        "first_alarm_issued": first_alarm_t.strftime("%Y-%m-%d") if first_alarm_t is not None else None,
        "event_day_reference": "2026-07-19 (Mon cloudburst, IST; regional reference)",
    }
    save_json(PUBLIC_DIR / nm["test"], doc)

    png_ok = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110)
        ax.plot(actual.index, actual, color="#0b0b0b", lw=1.8, label="GloFAS reanalysis (actual, modelled product)")
        ax.plot(actual.index, preds[1], color="#2a78d6", lw=1.4, ls="--",
                label=f"model v0 ({meta['shipped']}), 1 day ahead")
        ax.axhline(thr, color="#d03b3b", lw=1, ls=":", label=f"q90 threshold ({thr:.0f} m³/s)")
        ax.axvline(EVENT_DAY, color="#eb6834", lw=1, label="19 July (Mon cloudburst)")
        for d, a in alarms[1].items():
            if a:
                ax.plot([d], [preds[1][d]], "v", color="#d03b3b", ms=7)
        ax.set_title(f"{cfg['label']} - 2026 out-of-sample test (fitted through 2025)")
        ax.set_ylabel("discharge (m³/s)")
        ax.legend(fontsize=8, loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(PUBLIC_DIR / nm["test_png"])
        plt.close(fig)
        png_ok = True
    except Exception as err:  # noqa: BLE001
        print(f"  PNG skipped: {err}")

    # ---- section for REPLAY-FINDINGS ----
    S = []
    A = S.append
    A(f"## 7{'' if basin == 'dikhow' else 'c'}. Model v0 - 2026 true out-of-sample test"
      f"{'' if basin == 'dikhow' else f' ({cfg[chr(108) + chr(97) + chr(98) + chr(101) + chr(108)]})'}")
    A("")
    A(f"_Generated {now} by `backend/model/test_2026.py --basin {basin}`. Shipped model: "
      f"**{meta['shipped']}** ({meta['decision_statement']}). Fitted through 2025 only; "
      f"2026 was never seen._")
    A("")
    A(f"_{meta['target_caveat']}_")
    A("")
    A(f"- Threshold: {thr:.1f} m³/s (90th percentile, 2015-2025 training monsoons).")
    A(f"- Actual exceedance days in the window: {', '.join(doc['actual_exceed_days']) or 'none'}.")
    verdicts, n_ok = [], 0
    for dd_s in doc["alarm_days_h1"]:
        dd = pd.Timestamp(dd_s)
        ok = dd in actual.index and actual[dd] >= thr
        n_ok += bool(ok)
        verdicts.append(f"{dd_s} ({'correct' if ok else 'FALSE alarm'}, issued {(dd - pd.Timedelta(days=1)).date()})")
    A(f"- Model alarms: {'; '.join(verdicts) or 'none'} - {n_ok} correct, {len(verdicts) - n_ok} false.")
    for o in onsets:
        ld = leads[o]
        A(f"- Event onset {o.date()}: " + (f"first correct alarm {ld} day(s) before onset."
                                          if ld is not None and ld > 0 else
                                          "alarm only on the onset day itself (0 days' warning)."
                                          if ld == 0 else "NOT predicted (no alarm for this event)."))
    if onsets:
        any_lead = any((leads[o] or 0) > 0 for o in onsets)
        o0 = onsets[0]
        p0 = preds[1].get(o0)
        A(f"- **Bottom line: model v0 for the {cfg['label'].split(' (')[0]} "
          + (f"gave advance warning for at least one 2026 rise.**" if any_lead else
             f"gave no usable advance warning of the 2026 rises** - its prediction for the "
             f"first onset day ({o0.date()}) reached ~{p0:.0f} m³/s against the {thr:.0f} "
             f"threshold; correct alarms came only once the river was already high."))
    else:
        A(f"- No exceedance of the {thr:.0f} m³/s threshold occurred in this basin's 2026 "
          f"window: nothing to detect; {len(verdicts)} alarm(s) raised were false.")
    A("")
    A(f"Chart data: `public/{nm['test']}`" + (f"; rendered: `public/{nm['test_png']}`." if png_ok else "."))
    A("")
    (HIST_DIR / nm["section"]).write_text("\n".join(S) + "\n", encoding="utf-8")

    check = load_json(PUBLIC_DIR / nm["test"])
    assert len(check["dates"]) == len(check["actual_m3s"]) == len(check["predicted_h1_m3s"])
    print(f"OK {nm['test']} ({len(check['dates'])} days) {'+ PNG ' if png_ok else ''}+ section md")
    print(f"   [{basin}] {meta['shipped']}: events {[(str(o.date()), leads[o]) for o in onsets]}, "
          f"first alarm {first_alarm_t.date() if first_alarm_t is not None else 'never'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
