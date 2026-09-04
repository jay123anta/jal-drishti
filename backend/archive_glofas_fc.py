"""FORECAST SCOREBOARD (1/2) - archive each pipeline run's GloFAS forecast
and latest reanalysis per PoC river cell.

data/discharge.json is overwritten every run, so without this step no
GloFAS forecast could ever be scored against what later happened. Each run
appends:
  data/history/glofas/<cell>/forecasts.parquet  (issued_date, target_date,
      median/mean/min/max, discharge_m3s)  class FORECAST, archived:true
  data/history/glofas/<cell>/observed.parquet   (date, discharge_m3s)
      class OBSERVED (GloFAS reanalysis - a modelled product)
Deduplicated on (issued_date, target_date) / date; idempotent per day.
SIMULATED fixture points (live fetch degraded) are never archived.
"""

import pathlib
import sys

import pandas as pd

from common import DATA_DIR, FORECAST, OBSERVED, load_json, save_json, utc_now_iso

HIST = DATA_DIR / "history" / "glofas"
SRC_FC = ("GloFAS v4 ensemble forecast via Open-Meteo Flood API, archived per "
          "pipeline run with its issue date (retrieved forecast, not observation)")
SRC_OBS = ("GloFAS v4 consolidated reanalysis via Open-Meteo Flood API "
           "(MODELLED product, not observed river data)")


def merge(path: pathlib.Path, new: pd.DataFrame, keys: list) -> pd.DataFrame:
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    new = new.drop_duplicates(subset=keys, keep="last").sort_values(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new


def main() -> int:
    now = utc_now_iso()
    doc = load_json(DATA_DIR / "discharge.json")
    issued = doc["generated_at"][:10]
    n_fc = n_obs = 0
    for p in doc["points"]:
        fc = [{"issued_date": issued, "target_date": r["date"],
               "median_m3s": (r.get("ensemble") or {}).get("median", r["discharge_m3s"]),
               "mean_m3s": (r.get("ensemble") or {}).get("mean"),
               "min_m3s": (r.get("ensemble") or {}).get("min"),
               "max_m3s": (r.get("ensemble") or {}).get("max"),
               "discharge_m3s": r["discharge_m3s"]}
              for r in p["daily"] if r["class"] == FORECAST]
        obs = [{"date": r["date"], "discharge_m3s": r["discharge_m3s"]}
               for r in p["daily"] if r["class"] == OBSERVED]
        if not fc and not obs:
            continue                                   # degraded/simulated point
        d = HIST / p["id"]
        if fc:
            df = merge(d / "forecasts.parquet", pd.DataFrame(fc), ["issued_date", "target_date"])
            save_json(d / "forecasts.provenance.json", {
                "cell": p["id"], "rows": int(len(df)), "issue_dates": int(df["issued_date"].nunique()),
                "first_issued": str(df["issued_date"].min()), "last_issued": str(df["issued_date"].max()),
                "class": FORECAST, "archived": True, "source": SRC_FC, "retrieved_at": now})
            n_fc += len(fc)
        if obs:
            do = merge(d / "observed.parquet", pd.DataFrame(obs), ["date"])
            save_json(d / "observed.provenance.json", {
                "cell": p["id"], "rows": int(len(do)),
                "first": str(do["date"].min()), "last": str(do["date"].max()),
                "class": OBSERVED, "source": SRC_OBS, "retrieved_at": now})
            n_obs += len(obs)
    print(f"OK glofas archive: issue date {issued}, {n_fc} forecast rows + {n_obs} reanalysis "
          f"rows merged across {len(doc['points'])} cells -> data/history/glofas/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
