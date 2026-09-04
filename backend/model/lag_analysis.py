"""MODEL v0, STEP M2 (generalised in Phase 1) - lag-time analysis:
catchment rain -> downstream discharge, per basin.

    python backend/model/lag_analysis.py                 # dikhow (default)
    python backend/model/lag_analysis.py --basin kopili

Reads the M1 history partitions for the basin (backend/model/basins.py)
and answers, with numbers: how long after rain falls in the hills does
the discharge respond at the target cell?

HONESTY:
- Discharge is GloFAS v4 reanalysis - a MODELLED product, not observed river
  data (observed CWC gauge data will replace it when access is granted).
- GloFAS is DAILY, so lag resolution is capped at 1 day; hourly rain lets us
  bound the answer more tightly only through event timing, and we say so.

Method:
1. Cross-correlation at daily resolution: rolling catchment-mean rain sums
   (6/12/24/48 h ending at end-of-day d) vs discharge Q(d+lag) AND vs the
   day-over-day change dQ(d+lag), lags 0..5 days, monsoon (Jun-Sep) and
   all-year, 2015-2025. Uncertainty: leave-one-year-out spread of the
   best lag.
2. Event study: top 10 day-over-day discharge rises 2015-2025 (>= 7 days
   apart): time from the preceding 24 h catchment rain spike (hourly) to
   the discharge peak day. 2026 included as an OUT-OF-PERIOD check
   (partial test-only partition), never pooled with 2015-2025 stats.

Writes docs/LAG-ANALYSIS.md + data/history/lag_summary.json for dikhow,
docs/LAG-ANALYSIS-<basin>.md + data/history/lag_summary_<basin>.json
otherwise.
"""

import datetime
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import DATA_DIR, REPO_ROOT, save_json, utc_now_iso  # noqa: E402

from basins import BASINS, rain_point_ids  # noqa: E402

HIST_DIR = DATA_DIR / "history"
DOCS_DIR = REPO_ROOT / "docs"
YEARS = list(range(2015, 2026))
WINDOWS_H = [6, 12, 24, 48]
MAX_LAG_D = 5
TOP_EVENTS = 10
MIN_EVENT_SEP_D = 7

# Dikhow defaults (backward compatible for importers)
TARGET = BASINS["dikhow"]["target"]
RAIN_POINT_IDS = rain_point_ids("dikhow")


def lag_paths(basin: str) -> tuple[pathlib.Path, pathlib.Path]:
    if basin == "dikhow":
        return DOCS_DIR / "LAG-ANALYSIS.md", HIST_DIR / "lag_summary.json"
    return DOCS_DIR / f"LAG-ANALYSIS-{basin}.md", HIST_DIR / f"lag_summary_{basin}.json"


def load_hourly_rain(years, point_ids=None) -> pd.Series:
    """Catchment-mean hourly rain (unweighted mean of the anchor cells)."""
    frames = []
    for pid in (point_ids or RAIN_POINT_IDS):
        parts = []
        for y in years:
            pq = HIST_DIR / "rainfall" / pid / f"{y}.parquet"
            if pq.exists():
                parts.append(pd.read_parquet(pq))
        if not parts:
            continue
        df = pd.concat(parts)
        s = pd.Series(df["precipitation_mm"].to_numpy(),
                      index=pd.to_datetime(df["time"]), name=pid)
        frames.append(s)
    allp = pd.concat(frames, axis=1)
    return allp.mean(axis=1).sort_index()


def load_daily_q(pid: str, years) -> pd.Series:
    parts = []
    for y in years:
        pq = HIST_DIR / "discharge" / pid / f"{y}.parquet"
        if pq.exists():
            parts.append(pd.read_parquet(pq))
    df = pd.concat(parts)
    s = pd.Series(df["discharge_m3s"].to_numpy(),
                  index=pd.to_datetime(df["date"]), name=pid)
    return s.sort_index()


def daily_rain_sums(rain_h: pd.Series) -> pd.DataFrame:
    """R_w(d): rolling w-hour catchment rain sum ending at the END of day d."""
    out = {}
    for w in WINDOWS_H:
        roll = rain_h.rolling(w, min_periods=w).sum()
        eod = roll.groupby(roll.index.date).last()
        eod.index = pd.to_datetime(eod.index)
        out[f"rain_{w}h"] = eod
    return pd.DataFrame(out)


def xcorr_table(rain_d: pd.DataFrame, q: pd.Series, monsoon_only: bool,
                years=None) -> dict:
    q = q.copy()
    dq = q.diff()
    res = {}
    for col in rain_d.columns:
        res[col] = {}
        for lag in range(MAX_LAG_D + 1):
            r = rain_d[col]
            qq = q.shift(-lag)
            dqq = dq.shift(-lag)
            df = pd.DataFrame({"r": r, "q": qq, "dq": dqq}).dropna()
            if years is not None:
                df = df[df.index.year.isin(years)]
            if monsoon_only:
                df = df[(df.index.month >= 6) & (df.index.month <= 9)]
            res[col][lag] = (float(df["r"].corr(df["q"])),
                             float(df["r"].corr(df["dq"])),
                             int(len(df)))
    return res


def best_lag(table: dict, col: str, use_dq: bool) -> int:
    idx = 1 if use_dq else 0
    return max(table[col], key=lambda lag: table[col][lag][idx])


def loyo_spread(rain_d, q, col: str, use_dq: bool) -> list[int]:
    lags = []
    for leave in YEARS:
        keep = [y for y in YEARS if y != leave]
        t = xcorr_table(rain_d, q, monsoon_only=True, years=keep)
        lags.append(best_lag(t, col, use_dq))
    return lags


def find_events(q: pd.Series, top_n: int) -> list[dict]:
    dq = q.diff().dropna()
    order = dq.sort_values(ascending=False).index
    chosen = []
    for d in order:
        if len(chosen) >= top_n:
            break
        if any(abs((d - c).days) < MIN_EVENT_SEP_D for c in chosen):
            continue
        chosen.append(d)
    events = []
    for d in sorted(chosen):
        seg = q.loc[d:d + pd.Timedelta(days=4)]
        peak_day = seg.idxmax()
        events.append({"rise_day": d, "peak_day": peak_day,
                       "q_before": float(q.get(d - pd.Timedelta(days=1), np.nan)),
                       "q_peak": float(seg.max()),
                       "rise_m3s": float(dq[d])})
    return events


def rain_spike_before(rain_h: pd.Series, peak_day: pd.Timestamp) -> tuple:
    roll = rain_h.rolling(24, min_periods=24).sum()
    win = roll.loc[peak_day - pd.Timedelta(days=5): peak_day + pd.Timedelta(hours=23)]
    if win.empty:
        return None, None
    return win.idxmax(), float(win.max())


def analyze(basin: str) -> int:
    now = utc_now_iso()
    cfg = BASINS[basin]
    target = cfg["target"]
    pids = rain_point_ids(basin)
    md_path, json_path = lag_paths(basin)

    rain_h = load_hourly_rain(YEARS, pids)
    q = load_daily_q(target, YEARS)
    rain_d = daily_rain_sums(rain_h)
    t_mon = xcorr_table(rain_d, q, monsoon_only=True)
    t_all = xcorr_table(rain_d, q, monsoon_only=False)

    best = {}
    for col in rain_d.columns:
        bl_q = best_lag(t_mon, col, use_dq=False)
        bl_dq = best_lag(t_mon, col, use_dq=True)
        spread = loyo_spread(rain_d, q, col, use_dq=True)
        best[col] = {"lag_vs_q_days": bl_q, "lag_vs_dq_days": bl_dq,
                     "loyo_spread_dq": sorted(set(spread)),
                     "corr_dq_at_best": t_mon[col][bl_dq][1]}

    events = find_events(q, TOP_EVENTS)
    ev_rows = []
    for e in events:
        spike_t, mm24 = rain_spike_before(rain_h, e["peak_day"])
        lag_d = (e["peak_day"].date() - spike_t.date()).days if spike_t is not None else None
        ev_rows.append({**e, "rain_spike": spike_t, "rain_mm24": mm24, "lag_days": lag_d})

    rain26 = load_hourly_rain([2026], pids)
    q26 = load_daily_q(target, [2026])
    dq26 = q26.diff().dropna()
    d26 = dq26.idxmax()
    seg26 = q26.loc[d26:d26 + pd.Timedelta(days=4)]
    peak26 = seg26.idxmax()
    spike26, mm26 = rain_spike_before(rain26, peak26)
    lag26 = (peak26.date() - spike26.date()).days if spike26 is not None else None

    lags = [r["lag_days"] for r in ev_rows if r["lag_days"] is not None]
    lag_lo, lag_hi = int(min(lags)), int(max(lags))
    lag_med = int(np.median(lags))
    n_same = sum(1 for l in lags if l == 0)
    n_next = sum(1 for l in lags if l == 1)
    name = cfg["label"]
    if lag_med == 0:
        plain = (f"Rain in the {name.split(' (')[0]} hills reaches the {target} river data "
                 f"fast - usually the same day, sometimes the next day: typically within "
                 f"about 36 hours. ({n_same} of the {len(lags)} biggest rises registered "
                 f"same-day, {n_next} next-day; daily river data cannot resolve hours.)")
    else:
        hr_lo = max(lag_med * 24 - 12, 0)
        hr_hi = lag_hi * 24 + 12
        plain = (f"Rain in the {name.split(' (')[0]} hills typically shows up in the "
                 f"{target} river data {lag_med} day{'s' if lag_med != 1 else ''} later - "
                 f"roughly {hr_lo} to {hr_hi} hours (daily river data cannot resolve it "
                 f"more finely).")

    DOCS_DIR.mkdir(exist_ok=True)
    L = []
    A = L.append
    A(f"# LAG-ANALYSIS - {name}: catchment rain vs downstream discharge")
    A("")
    A(f"Generated {now} by `backend/model/lag_analysis.py --basin {basin}` from the")
    A(f"history (2015-2025; 2026 used only as an out-of-period check). Target cell:")
    A(f"`{target}`; rain anchors: {', '.join(pids)} (CATCHMENT.md).")
    A("")
    A("**Target caveat: discharge is GloFAS v4 reanalysis - a MODELLED")
    A("product, not observed river data. Observed CWC gauge data will")
    A("replace this when access is granted.** GloFAS is daily, so all lags")
    A("here have 1-day resolution; hour ranges below are day-boundary bounds,")
    A("not measured hours.")
    A("")
    A("## Plain-language summary")
    A("")
    A(f"**{plain}**")
    A("")
    A("## 1. Cross-correlation (monsoon Jun-Sep, 2015-2025)")
    A("")
    A("| rain window | best lag vs Q | best lag vs dQ | corr(dQ) at best | LOYO spread (dQ) |")
    A("|---|---|---|---|---|")
    for col, b in best.items():
        A(f"| {col} | {b['lag_vs_q_days']} d | {b['lag_vs_dq_days']} d "
          f"| {b['corr_dq_at_best']:.3f} | {b['loyo_spread_dq']} d |")
    A("")
    A("Full monsoon correlation table (corr vs dQ):")
    A("")
    A("| window \\ lag | " + " | ".join(f"{lag} d" for lag in range(MAX_LAG_D + 1)) + " |")
    A("|---|" + "---|" * (MAX_LAG_D + 1))
    for col in rain_d.columns:
        A(f"| {col} | " + " | ".join(f"{t_mon[col][lag][1]:.3f}"
                                     for lag in range(MAX_LAG_D + 1)) + " |")
    A("")
    A("All-year best lags (context): " +
      ", ".join(f"{col}: {best_lag(t_all, col, True)} d vs dQ" for col in rain_d.columns))
    A("")
    A("## 2. Event study - top 10 discharge rises, 2015-2025")
    A("")
    A("| # | rise day | peak day | Q before -> peak (m³/s) | rain spike (24 h sum) | lag (days) |")
    A("|---|---|---|---|---|---|")
    for i, r in enumerate(ev_rows, 1):
        spike = (r["rain_spike"].strftime("%Y-%m-%d %H:00Z")
                 if r["rain_spike"] is not None else "none found")
        A(f"| {i} | {r['rise_day'].date()} | {r['peak_day'].date()} "
          f"| {r['q_before']:.0f} -> {r['q_peak']:.0f} "
          f"| {spike} ({r['rain_mm24']:.0f} mm) | {r['lag_days']} |")
    A("")
    A(f"Median event lag: **{lag_med} day(s)**; range {lag_lo}-{lag_hi} days")
    A(f"across the {len(lags)} events with an identifiable rain spike.")
    A("")
    A("## 3. 2026 out-of-period check (never pooled with the above)")
    A("")
    A(f"Largest 2026 rise: {d26.date()} (+{dq26.max():.0f} m³/s), peak "
      f"{peak26.date()} at {q26.max():.0f} m³/s; preceding 24 h rain spike "
      f"{spike26.strftime('%Y-%m-%d %H:00Z')} ({mm26:.0f} mm) -> lag "
      f"**{lag26} day(s)**.")
    A("")
    A("## Honesty notes")
    A("")
    A("- Catchment rain is the unweighted mean of the anchor cells")
    A("  (CATCHMENT.md); no drainage-area weighting - documented limitation.")
    A("- Correlations are against a modelled discharge product; GloFAS itself")
    A("  ingests precipitation, so rain-discharge correlation is partly the")
    A("  GloFAS model's own routing, not independent confirmation of nature.")
    A("- 1-day lag at daily resolution means anywhere from ~1 to ~47 hours in")
    A("  reality; the hour range in the summary is that bound, not a")
    A("  measurement.")
    A("")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")

    save_json(json_path, {
        "generated_at": now, "basin": basin, "target": target,
        "class": "OBSERVED",
        "source": ("derived by backend/model/lag_analysis.py from ERA5-family "
                   "rain + GloFAS reanalysis discharge (modelled products)"),
        "retrieved_at": now,
        "plain_sentence": plain,
        "best_lags_monsoon": best,
        "event_lags_days": lags,
        "median_event_lag_days": lag_med,
        "out_of_period_2026_lag_days": lag26,
        "july_2026_out_of_period_lag_days": lag26,
    })
    print(f"OK {md_path.name} + {json_path.name}")
    print(f"   {plain}")
    print(f"   events lag range {lag_lo}-{lag_hi} d (median {lag_med}); 2026 check: {lag26} d")
    return 0


def main() -> int:
    args = sys.argv[1:]
    basin = args[args.index("--basin") + 1] if "--basin" in args else "dikhow"
    if basin not in BASINS:
        raise SystemExit(f"unknown basin {basin!r}; known: {list(BASINS)}")
    return analyze(basin)


if __name__ == "__main__":
    sys.exit(main())
