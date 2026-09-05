"""MODEL v1, STEP V1c - July 2026 what-if analysis of the NOT-shipped v1
candidate (analysis only; model v0 remains the shipped model per the
pre-registered V1b decision).

Question: with the rain forecasts ACTUALLY ISSUED at the time (Previous
Runs archive), would the v1 XGB candidate have alarmed earlier on the
19-20 July 2026 event than shipped v0 did (v0: zero lead)? And what did
the issued forecasts say for 19 and 20 July?

Deterministic rebuild of the gated candidate (same params/seed/data as
train_v1.py). Writes data/history/model_2026_v1_section.md, which
analyze_replay.py appends to REPLAY-FINDINGS.md. Run once and committed;
not a pipeline step (depends on training-time artifacts).

Target caveat: GloFAS v4 reanalysis - a MODELLED product, not observed
river data; observed CWC gauge data will replace this when access is
granted.
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import load_json, utc_now_iso  # noqa: E402

from mdata import build_daily_frame, feature_frame  # noqa: E402
from train_v1 import (H, YEARS, actual_next24, add_rainnext_feature,  # noqa: E402
                      fit_ml, load_fc_daily, pred_for_days)
from fetch_history import HIST_DIR  # noqa: E402
from test_2026 import MODELS_DIR  # noqa: E402

THR_META = "dikhow_v0_meta.json"
WINDOW = (pd.Timestamp("2026-06-15"), pd.Timestamp("2026-08-05"))
EVENT_ONSETS = [pd.Timestamp("2026-07-16"), pd.Timestamp("2026-07-20")]


def main() -> int:
    now = utc_now_iso()
    thr = load_json(MODELS_DIR / THR_META)["threshold_q90_monsoon_2015_2025_m3s"]
    v1m = load_json(MODELS_DIR / "dikhow_v1_meta.json")
    gated = load_json(HIST_DIR / "model_v1_metrics.json")["ship_decision"]["gated_candidate"]

    # train exactly as V1b did (perfect-prog, 2015-2025, same seed)
    base_tr = build_daily_frame(YEARS)
    daily_pp = add_rainnext_feature(base_tr, actual_next24(YEARS))
    feats_pp = feature_frame(daily_pp)
    feats_pp["rain_next24"] = daily_pp["rain_next24"]
    model, cols = fit_ml(feats_pp, YEARS, gated)

    # 2026 operational frame: real issued forecasts
    base26 = build_daily_frame([2026])
    fc = load_fc_daily()
    daily_op26 = add_rainnext_feature(base26, fc["fc1_target"])
    feats26 = feature_frame(daily_op26)
    feats26["rain_next24"] = daily_op26["rain_next24"]

    lo, hi = WINDOW
    pred = pred_for_days(model, feats26, cols).reindex(base26.index).loc[lo:hi]
    actual = base26["q"].loc[lo:hi]
    alarms = pred >= thr

    act24 = actual_next24([2026])
    fc_19 = fc["fc1_target"].get(pd.Timestamp("2026-07-19"))
    fc_20 = fc["fc1_target"].get(pd.Timestamp("2026-07-20"))
    act_19 = act24.get(pd.Timestamp("2026-07-19"))
    act_20 = act24.get(pd.Timestamp("2026-07-20"))

    leads = {}
    for o in EVENT_ONSETS:
        run = []
        d = o
        while d in actual.index and actual[d] >= thr:
            run.append(d)
            d += pd.Timedelta(days=1)
        alarm_times = [dd - pd.Timedelta(days=H) for dd in run
                       if dd in alarms.index and alarms[dd]]
        leads[o] = (o - min(alarm_times)).days if alarm_times else None

    alarm_days = [d.strftime("%Y-%m-%d") for d, a in alarms.items() if a]

    S = []
    A = S.append
    A("### 7b. Model v1 what-if (analysis only - v1 was NOT shipped)")
    A("")
    A(f"_Generated {now} by `backend/model/analysis_2026_v1.py`. The v1 "
      f"attempt (adding tomorrow's rain as actually forecast at the time) "
      f"FAILED its pre-registered gate - {v1m['decision_statement']} - so "
      f"model v0 remains shipped. This section asks only: would the v1 "
      f"candidate ({gated.upper()}) have done better on this one event?_")
    A("")
    A(f"- What the issued forecasts said: for 19 July, "
      f"{fc_19:.0f} mm was forecast the day before (actual: {act_19:.0f} mm); "
      f"for 20 July, {fc_20:.0f} mm (actual: {act_20:.0f} mm). "
      + ("The forecasts saw meaningful rain coming."
         if (fc_19 or 0) > 20 else
         "The issued forecast largely missed the burst - the weather model "
         "under-predicted it."))
    A(f"- v1-candidate alarm days (target day predicted >= {thr:.0f} m³/s): "
      f"{', '.join(alarm_days) or 'none'}.")
    for o in EVENT_ONSETS:
        ld = leads[o]
        A(f"- Event onset {o.date()}: " +
          (f"first correct alarm {ld} day(s) before onset."
           if ld is not None and ld > 0 else
           "alarm only on the onset day itself (0 days' warning)."
           if ld == 0 else "NOT predicted."))
    v0_leads = {"2026-07-16": None, "2026-07-20": 0}   # from section 7
    verdict_better = any((leads[o] or -1) > (v0_leads[str(o.date())] if v0_leads[str(o.date())] is not None else -1)
                         for o in EVENT_ONSETS)
    A(f"- **Verdict: the v1 candidate "
      f"{'would have improved on' if verdict_better else 'does NOT improve on'} "
      f"shipped v0 for this event**"
      + (" - consistent with the walk-forward decision to keep v0." if not verdict_better else
         " on this single event - but it still failed the two-season gate, "
         "and one event does not overturn a pre-registered rule."))
    p20 = pred.get(pd.Timestamp("2026-07-20"))
    A("")
    A(f"Why rain foresight did not become lead time: the burst was forecast "
      f"for 19 July itself, but the Dikhow's response appears in the DAILY "
      f"GloFAS series on the 20th. At prediction time on the 19th the "
      f"candidate saw both the fallen rain and the (modest, {fc_20:.0f} mm) "
      f"forecast for the 20th, and still lifted its prediction for the 20th "
      f"only to {p20:.0f} m³/s against the {thr:.0f} threshold - learned "
      f"daily-step responses are damped. Combined with the perfect-prog "
      f"bound in the v1 card (~4-10% MAE ceiling), the evidence now points "
      f"one way: the binding constraint is the daily, modelled target. "
      f"Sub-daily OBSERVED river data (CWC gauges) is what would let rain "
      f"foresight become warning time.")
    A("")
    (HIST_DIR / "model_2026_v1_section.md").write_text("\n".join(S) + "\n",
                                                       encoding="utf-8")
    print("OK data/history/model_2026_v1_section.md")
    print(f"   fc(19 Jul)={fc_19:.0f}mm vs actual {act_19:.0f}mm; "
          f"fc(20 Jul)={fc_20:.0f}mm vs actual {act_20:.0f}mm")
    print(f"   v1 leads: {[(str(o.date()), leads[o]) for o in EVENT_ONSETS]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
