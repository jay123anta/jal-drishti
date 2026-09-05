"""Three-way rain validation: Open-Meteo vs IMERG vs IMD, at every anchor.

The whole model runs on Open-Meteo rain. Two references now exist for the
same points and days:
  - IMD Pune 0.25-deg gridded daily rainfall  (gauge-based; Indian
    hydrology's ground truth for daily rain)
  - NASA GPM IMERG Final daily satellite rain (independent, space-based)

This scores all three against each other over their overlapping history,
per anchor, so we can see not just "does the model agree with IMD" but
"where the two references themselves disagree" - which tells us how much
of any model-vs-IMD gap is real model error versus reference uncertainty.

Method (fixed before results, thresholds identical to imd_rain_check.py so
nothing is tuned):
  - Anchors: every basin rain point (backend/model/basins.py) that has BOTH
    archived Open-Meteo history AND an IMERG series AND an IMD grid cell.
  - Series: IMD native daily; Open-Meteo summed over the IMD observation
    day (03:00 UTC bins, reusing imd_rain_check.om_daily_imd_aligned);
    IMERG native daily.
  - Compared on the intersection of dates all three cover, >= 300 days.
  - Metrics per pair (all days + monsoon Jun-Sep): Pearson r, each side's
    mean daily mm, bias (first/second), daily MAE.

HONEST CAVEAT (disclosed, not fudged): IMD's day runs 08:30 IST -> 08:30
IST and Open-Meteo is summed to match it; IMERG's daily GIS product is a
00-24 UTC accumulation. So the two IMERG-involving pairs carry a ~3-hour
day-boundary offset against IMD/OM - minor at monsoon daily scale but real,
and it slightly depresses IMERG correlations. It is NOT corrected here
because the IMERG daily product cannot be re-windowed after aggregation.

Outputs: data/history/rain_validation_threeway.json + docs/RAIN-VALIDATION-THREEWAY.md
"""
from __future__ import annotations
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))
from imd_rain_check import anchors, imd_frame, om_daily_imd_aligned  # noqa: E402

IMERG_DIR = BASE_DIR / "data" / "history" / "imerg"
OUT_JSON = BASE_DIR / "data" / "history" / "rain_validation_threeway.json"
OUT_MD = BASE_DIR / "docs" / "RAIN-VALIDATION-THREEWAY.md"
MIN_DAYS = 300
SRC = ("Open-Meteo archive rain (model) vs NASA GPM IMERG Final daily "
       "satellite rain vs IMD Pune 0.25-deg gridded daily rainfall")


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def imerg_daily(pid: str) -> pd.Series | None:
    p = IMERG_DIR / f"{pid}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty or "rain_mm" not in df:
        return None
    s = pd.Series(pd.to_numeric(df["rain_mm"], errors="coerce").values,
                  index=pd.to_datetime(df["date"]).dt.date).sort_index().dropna()
    return s[~s.index.duplicated(keep="first")]


def pair(sel: pd.DataFrame, a: str, b: str) -> dict:
    x, y = sel[a], sel[b]
    return {"r": round(float(x.corr(y)), 3),
            f"{a}_mean": round(float(x.mean()), 2),
            f"{b}_mean": round(float(y.mean()), 2),
            "bias": round(float(x.mean() / y.mean()), 2) if y.mean() > 0 else None,
            "mae": round(float((x - y).abs().mean()), 2)}


def score(a: dict, imd: pd.DataFrame) -> dict:
    glat, glon = round(a["lat"] * 4) / 4, round(a["lon"] * 4) / 4
    cell = imd[(imd["lat"] == glat) & (imd["lon"] == glon)]
    if len(cell) < MIN_DAYS:
        return {"skipped": f"IMD cell ({glat},{glon}) has {len(cell)} days (outside coverage)"}
    iv = cell.set_index("date")["rain_mm"]
    iv.index = pd.to_datetime(iv.index).date
    om = om_daily_imd_aligned(a["id"])
    if om is None:
        return {"skipped": "no Open-Meteo history archived"}
    im = imerg_daily(a["id"])
    if im is None:
        return {"skipped": "no IMERG series archived"}
    both = pd.DataFrame({"imd": iv, "om": om, "imerg": im}).dropna()
    if len(both) < MIN_DAYS:
        return {"skipped": f"only {len(both)} days covered by all three"}
    idx_month = pd.Series(pd.to_datetime(both.index)).dt.month
    res = {"imd_cell": [glat, glon], "n_days": int(len(both))}
    for tag, sel in (("all", both),
                     ("monsoon", both[idx_month.isin([6, 7, 8, 9]).values])):
        if len(sel) < 100:
            continue
        res[tag] = {"n": int(len(sel)),
                    "om_vs_imd": pair(sel, "om", "imd"),
                    "imerg_vs_imd": pair(sel, "imerg", "imd"),
                    "om_vs_imerg": pair(sel, "om", "imerg")}
    return res


def med(rows, tag, pairkey, key):
    vals = [r[tag][pairkey][key] for r in rows
            if tag in r and r[tag].get(pairkey, {}).get(key) is not None]
    return round(float(np.median(vals)), 3) if vals else None


def main() -> int:
    now = utc_now_iso()
    imd = imd_frame()
    print(f"IMD grid: {len(imd)} cell-days; scoring three-way at each anchor...")
    results, skipped = [], []
    for a in anchors():
        try:
            sc = score(a, imd)
        except Exception as err:  # noqa: BLE001
            sc = {"skipped": f"error: {err}"}
        rec = {"id": a["id"], "basin": a["basin"], **sc}
        (skipped if "skipped" in sc else results).append(rec)
        if "skipped" in sc:
            print(f"  {a['id']:20s} skip: {sc['skipped']}")
        else:
            m = sc.get("monsoon", {})
            print(f"  {a['id']:20s} monsoon r: OM-IMD={m.get('om_vs_imd',{}).get('r')} "
                  f"IMERG-IMD={m.get('imerg_vs_imd',{}).get('r')} "
                  f"OM-IMERG={m.get('om_vs_imerg',{}).get('r')}")

    summary = {"anchors_scored": len(results), "anchors_skipped": len(skipped)}
    for pk in ("om_vs_imd", "imerg_vs_imd", "om_vs_imerg"):
        summary[pk] = {"monsoon_median_r": med(results, "monsoon", pk, "r"),
                       "monsoon_median_bias": med(results, "monsoon", pk, "bias"),
                       "monsoon_median_mae": med(results, "monsoon", pk, "mae")}
    OUT_JSON.write_text(json.dumps({
        "generated_at": now, "source": SRC, "class": "OBSERVED",
        "method": {"min_days": MIN_DAYS,
                   "day_alignment": "IMD/OM on IMD 08:30 IST day; IMERG native 00-24 UTC day (~3h offset, disclosed)"},
        "summary": summary, "anchors": results, "skipped": skipped,
    }, indent=1), encoding="utf-8")

    def row(pk):
        s = summary[pk]
        return f"| {pk.replace('_',' ')} | {s['monsoon_median_r']} | {s['monsoon_median_bias']} | {s['monsoon_median_mae']} |"
    md = [
        "# RAIN-VALIDATION-THREEWAY - Open-Meteo vs IMERG vs IMD\n",
        f"Generated {now} by `backend/three_way_rain.py`. Thresholds identical to",
        "`imd_rain_check.py`; nothing tuned. Three daily-rain sources scored against",
        "each other at every basin rain anchor over their overlapping history.\n",
        "Day alignment: IMD day = 08:30 IST -> 08:30 IST; Open-Meteo summed to match;",
        "IMERG is a 00-24 UTC daily accumulation (a ~3h boundary offset that slightly",
        "lowers IMERG correlations, disclosed, not corrected - the daily product cannot",
        "be re-windowed).\n",
        f"## Summary (monsoon Jun-Sep, medians across {summary['anchors_scored']} anchors, "
        f"{summary['anchors_skipped']} skipped)\n",
        "| pair | median corr r | median bias (first/second) | median daily MAE mm |",
        "|---|---|---|---|",
        row("om_vs_imd"), row("imerg_vs_imd"), row("om_vs_imerg"),
        "\n## Per-anchor (monsoon corr r)\n",
        "| anchor | basin | days | OM-IMD | IMERG-IMD | OM-IMERG |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: -(x.get("monsoon", {}).get("om_vs_imd", {}).get("r") or 0)):
        m = r.get("monsoon", {})
        md.append(f"| {r['id']} | {r['basin']} | {m.get('n')} "
                  f"| {m.get('om_vs_imd',{}).get('r')} | {m.get('imerg_vs_imd',{}).get('r')} "
                  f"| {m.get('om_vs_imerg',{}).get('r')} |")
    if skipped:
        md += ["\n## Skipped anchors\n"] + [f"- {r['id']} ({r['basin']}): {r['skipped']}" for r in skipped]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"OK three-way rain: {len(results)} anchors scored, {len(skipped)} skipped "
          f"-> {OUT_JSON.name} + {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
