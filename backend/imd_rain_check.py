"""THE rain validation: Open-Meteo model rain vs IMD's gridded gauge analysis.

The NWDP telemetry-gauge attempt ended honestly inconclusive (the gauge
feed was unusable). This is the proper reference: IMD Pune's 0.25-deg
gridded daily rainfall, interpolated from IMD's own gauge network - the
dataset Indian hydrology treats as ground truth for daily rain.

Method (fixed before results were computed; thresholds identical to the
earlier attempt so nothing is tuned):
- Points: every rain anchor of every basin (backend/model/basins.py) that
  has archived hourly Open-Meteo history.
- Day alignment: an IMD daily value dated D covers 08:30 IST of D-1 to
  08:30 IST of D. Open-Meteo hourly (UTC) is summed over
  [D-1 03:00 UTC, D 03:00 UTC) to match - stated, not fudged.
- IMD cell: the nearest 0.25-deg grid point to the anchor.
- Years 2015-2024 (2025 partial ok); metrics on ALL days and monsoon
  (Jun-Sep) separately: Pearson r, mean daily mm both sides, bias
  OM/IMD, MAE; heavy day = IMD >= 25 mm -> detected strict (OM >= 25) /
  lenient (OM >= 10); false alarm = OM >= 25 while IMD < 10.

Outputs: data/history/rain_validation_imd.json +
docs/RAIN-VALIDATION-IMD.md. Anchors without IMD coverage are listed as
skipped with the reason.
"""
from __future__ import annotations
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend" / "model"))
from basins import BASINS  # noqa: E402

GRID_DIR = BASE_DIR / "data" / "history" / "imd" / "grid_ne"
RAIN_DIR = BASE_DIR / "data" / "history" / "rainfall"
OUT_JSON = BASE_DIR / "data" / "history" / "rain_validation_imd.json"
OUT_MD = BASE_DIR / "docs" / "RAIN-VALIDATION-IMD.md"

HEAVY, LENIENT = 25.0, 10.0
SRC = ("Open-Meteo archive rain (model) scored against IMD Pune 0.25-deg "
       "gridded daily rainfall (gauge-based analysis)")


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def imd_frame() -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(GRID_DIR.glob("*.parquet"))]
    if not parts:
        raise SystemExit("no IMD grid parquets - run fetch_imd_grid.py first")
    return pd.concat(parts, ignore_index=True)


def om_daily_imd_aligned(pid: str) -> pd.Series | None:
    """Open-Meteo hourly summed over the IMD observation day (03:00 UTC bins)."""
    pdir = RAIN_DIR / pid
    if not pdir.exists():
        return None
    parts = [pd.read_parquet(f) for f in sorted(pdir.glob("*.parquet"))]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    t = pd.to_datetime(df["time"])
    v = pd.to_numeric(df["precipitation_mm"], errors="coerce")
    s = pd.Series(v.values, index=t).sort_index().dropna()
    # shift so that [D-1 03:00, D 03:00) lands on date D
    shifted = s.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=21)
    return shifted.groupby(shifted.index.date).sum()


def anchors() -> list[dict]:
    seen, out = set(), []
    for b, cfg in BASINS.items():
        for pt in cfg["rain_points"]:
            if pt["id"] in seen:
                continue
            seen.add(pt["id"])
            out.append({"id": pt["id"], "lat": pt["lat"], "lon": pt["lon"], "basin": b})
    return out


def score(a: dict, imd: pd.DataFrame) -> dict:
    glat = round(a["lat"] * 4) / 4
    glon = round(a["lon"] * 4) / 4
    cell = imd[(imd["lat"] == glat) & (imd["lon"] == glon)]
    if len(cell) < 300:
        return {"skipped": f"IMD cell ({glat},{glon}) has {len(cell)} days "
                           "(outside gauge coverage - common for border hills)"}
    iv = cell.set_index("date")["rain_mm"]
    iv.index = pd.to_datetime(iv.index).date
    om = om_daily_imd_aligned(a["id"])
    if om is None:
        return {"skipped": "no Open-Meteo history archived for this anchor"}
    both = pd.DataFrame({"imd": iv, "om": om}).dropna()
    if len(both) < 300:
        return {"skipped": f"only {len(both)} overlapping days"}
    res = {"imd_cell": [glat, glon], "n_days": int(len(both))}
    for tag, sel in (("all", both),
                     ("monsoon", both[pd.Series(pd.to_datetime(both.index)).dt.month
                                      .isin([6, 7, 8, 9]).values])):
        if len(sel) < 100:
            continue
        heavy = sel[sel["imd"] >= HEAVY]
        fa = sel[(sel["om"] >= HEAVY) & (sel["imd"] < LENIENT)]
        res[tag] = {
            "n": int(len(sel)),
            "r": round(float(sel["imd"].corr(sel["om"])), 3),
            "imd_mean": round(float(sel["imd"].mean()), 2),
            "om_mean": round(float(sel["om"].mean()), 2),
            "bias": round(float(sel["om"].mean() / sel["imd"].mean()), 2)
            if sel["imd"].mean() > 0 else None,
            "mae": round(float((sel["imd"] - sel["om"]).abs().mean()), 2),
            "heavy": int(len(heavy)),
            "pod25": round(float((heavy["om"] >= HEAVY).mean()), 2) if len(heavy) else None,
            "pod10": round(float((heavy["om"] >= LENIENT).mean()), 2) if len(heavy) else None,
            "fa_days": int(len(fa)),
        }
    return res


def med(rows, tag, key):
    vals = [r[tag][key] for r in rows if tag in r and r[tag].get(key) is not None]
    return round(float(np.median(vals)), 3) if vals else None


def main() -> int:
    now = utc_now_iso()
    imd = imd_frame()
    print(f"IMD grid: {len(imd)} cell-days loaded; scoring anchors...")
    results, skipped = [], []
    for a in anchors():
        try:
            sc = score(a, imd)
        except Exception as err:  # noqa: BLE001
            sc = {"skipped": f"error: {err}"}
        rec = {"id": a["id"], "basin": a["basin"], **sc}
        (skipped if "skipped" in sc else results).append(rec)
        tag = sc.get("skipped") or (
            f"monsoon r={sc.get('monsoon', {}).get('r')} bias={sc.get('monsoon', {}).get('bias')} "
            f"pod10={sc.get('monsoon', {}).get('pod10')}")
        print(f"  {a['id']:20s} {tag}")

    summary = {
        "anchors_scored": len(results), "anchors_skipped": len(skipped),
        "monsoon_median_r": med(results, "monsoon", "r"),
        "monsoon_median_bias": med(results, "monsoon", "bias"),
        "monsoon_median_mae": med(results, "monsoon", "mae"),
        "monsoon_median_pod25": med(results, "monsoon", "pod25"),
        "monsoon_median_pod10": med(results, "monsoon", "pod10"),
        "heavy_days_total": sum(r["monsoon"]["heavy"] for r in results if "monsoon" in r),
        "fa_days_total": sum(r["monsoon"]["fa_days"] for r in results if "monsoon" in r),
    }
    OUT_JSON.write_text(json.dumps({
        "generated_at": now, "source": SRC, "class": "OBSERVED",
        "method": {"heavy_mm": HEAVY, "lenient_mm": LENIENT,
                   "day_alignment": "IMD day D = 03:00 UTC D-1 .. 03:00 UTC D"},
        "summary": summary, "anchors": results, "skipped": skipped,
    }, indent=1), encoding="utf-8")

    md = [
        "# RAIN-VALIDATION-IMD - Open-Meteo vs IMD gridded rainfall\n",
        f"Generated {now} by `backend/imd_rain_check.py`. Thresholds identical",
        "to the earlier gauge attempt (docs/RAIN-VALIDATION.md) - nothing tuned.",
        "Reference: IMD Pune 0.25-deg gridded daily rainfall - the gauge-based",
        "analysis Indian hydrology treats as ground truth. Day alignment: an",
        "IMD day covers 08:30 IST to 08:30 IST; Open-Meteo hours are summed",
        "over exactly that window.\n",
        "## Summary (monsoon Jun-Sep, medians across anchors)\n",
        "| anchors scored | corr r | bias OM/IMD | daily MAE mm | heavy-day POD (>=25) | POD (>=10) | false alarms |",
        "|---|---|---|---|---|---|---|",
        f"| {summary['anchors_scored']} ({summary['anchors_skipped']} skipped) "
        f"| {summary['monsoon_median_r']} | {summary['monsoon_median_bias']} "
        f"| {summary['monsoon_median_mae']} | {summary['monsoon_median_pod25']} "
        f"| {summary['monsoon_median_pod10']} "
        f"| {summary['fa_days_total']} across {summary['heavy_days_total']} heavy days |",
        "\n## Per-anchor (monsoon)\n",
        "| anchor | basin | IMD cell | days | r | bias | MAE | heavy | POD25 | POD10 | FA |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: -(x.get("monsoon", {}).get("r") or 0)):
        m = r.get("monsoon", {})
        md.append(f"| {r['id']} | {r['basin']} | {r['imd_cell']} | {m.get('n')} "
                  f"| {m.get('r')} | {m.get('bias')} | {m.get('mae')} | {m.get('heavy')} "
                  f"| {m.get('pod25')} | {m.get('pod10')} | {m.get('fa_days')} |")
    if skipped:
        md += ["\n## Skipped anchors\n"]
        md += [f"- {r['id']} ({r['basin']}): {r['skipped']}" for r in skipped]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"OK imd rain check: {len(results)} anchors scored, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
