"""PHASE 1 (AI-ready platform) - input DRIFT monitor, generic per basin.

    python backend/model/drift.py                  # dikhow
    python backend/model/drift.py --basin kopili

Compares the basin's live model inputs (from data/<forecast>: discharge
now, catchment rain 24 h / 48 h) against the 2015-2025 training
distribution for the same time of year (+/- 15 days of day-of-year) and
for the monsoon overall. Writes public/<drift> and never changes a
colour: an input outside the training envelope means the model is
extrapolating, and the Technical view says so.

Status per input: in-range (p01..p99 of the seasonal reference), extreme
(beyond p99 / below p01 but within the all-time min..max), out-of-range
(beyond anything seen 2015-2025). Overall = worst input.
"""

import datetime
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import DATA_DIR, PUBLIC_DIR, load_json, save_json, utc_now_iso  # noqa: E402

from basins import BASINS, model_names  # noqa: E402
from mdata import build_daily_frame, monsoon_mask  # noqa: E402

YEARS = list(range(2015, 2026))
INPUTS = {"q_today_m3s": "q", "catchment_rain_24h_mm": "rain_24h",
          "catchment_rain_48h_mm": "rain_48h"}


def status_of(v, ref: np.ndarray):
    ref = ref[~np.isnan(ref)]
    if len(ref) < 30 or v is None:
        return "unknown", None
    pct = float((ref <= v).mean())
    lo, hi = np.percentile(ref, [1, 99])
    if v > ref.max() or v < ref.min():
        return "out-of-range", pct
    if v > hi or v < lo:
        return "extreme", pct
    return "in-range", pct


def main() -> int:
    args = sys.argv[1:]
    basin = args[args.index("--basin") + 1] if "--basin" in args else "dikhow"
    nm = model_names(basin)
    now = utc_now_iso()
    out = {"generated_at": now, "basin": basin, "model": nm["method"], "inputs": {},
           "overall": "unknown",
           "reference": "2015-2025 GloFAS/ERA5-family history; seasonal = same day-of-year "
                        "+/- 15 d across years; monsoon = Jun-Sep all years",
           "note": "drift never changes a colour; it flags extrapolation in the Technical view"}
    try:
        mf = load_json(DATA_DIR / nm["forecast"])
        assert not mf.get("degraded")
    except (FileNotFoundError, ValueError, AssertionError):
        out["overall"] = "degraded"
        out["error"] = f"{nm['forecast']} missing or degraded - no live inputs to compare"
        save_json(PUBLIC_DIR / nm["drift"], out)
        print(f"  [{basin}] drift: DEGRADED ({out['error']})")
        return 0

    daily = build_daily_frame(YEARS, basin)
    doy = datetime.datetime.now(datetime.timezone.utc).timetuple().tm_yday
    d = daily.index.dayofyear.to_numpy()
    dd = np.minimum(np.abs(d - doy), 366 - np.abs(d - doy))
    seasonal, monsoon = dd <= 15, monsoon_mask(daily.index)
    order = {"in-range": 0, "unknown": 0, "extreme": 1, "out-of-range": 2}
    worst = "in-range"
    for key, col in INPUTS.items():
        v = mf["inputs"][key]["value"]
        st_s, pct_s = status_of(v, daily[col].to_numpy()[seasonal])
        st_m, pct_m = status_of(v, daily[col].to_numpy()[monsoon])
        out["inputs"][key] = {
            "value": v, "class": mf["inputs"][key]["class"], "source": mf["inputs"][key]["source"],
            "retrieved_at": now,
            "seasonal_status": st_s, "seasonal_percentile": round(pct_s, 3) if pct_s is not None else None,
            "monsoon_status": st_m, "monsoon_percentile": round(pct_m, 3) if pct_m is not None else None,
        }
        if order[st_s] > order[worst]:
            worst = st_s
    out["overall"] = worst
    save_json(PUBLIC_DIR / nm["drift"], out)
    print(f"  [{basin}] drift: " + ", ".join(
        f"{k}={i['seasonal_status']} (p{int(100 * (i['seasonal_percentile'] or 0))})"
        for k, i in out["inputs"].items()) + f" -> overall {worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
