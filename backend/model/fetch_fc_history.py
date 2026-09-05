"""MODEL v1, STEP V1a - backfill ARCHIVED rain forecasts for the Dikhow
catchment (the input whose absence made model v0 fail).

Source: Open-Meteo Previous Runs API (previous-runs-api.open-meteo.com,
free, no key): hourly `precipitation_previous_day1` / `_previous_day2` =
precipitation at each hour AS FORECAST by the model run issued ~1/2 days
earlier. Summing a target day's hours gives "that day's rain as it was
forecast 1/2 days ahead" - the leakage-safe feature (issuance timing rests
on the API's documented contract; a documented project rule).

Coverage (probed, recorded in internal records): non-null from Jun 2024 only.
Window fetched: 2024-06-01 .. 2026-08-05.

Provenance: these are retrieved PAST FORECASTS -> class FORECAST with
archived: true in the sidecar, source naming the exact product.

Storage mirrors M1: data/history/rainfall_fc/<point>/<year>.parquet +
provenance sidecar; raw gz cache; BACKFILL-STATE.json section
"rainfall_fc". Idempotent, resumable, rate-limited. Exit 1 if pending
partitions remain (rerun to resume).
"""

import gzip
import json
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import fetch_json, save_failed_request, save_json, utc_now_iso  # noqa: E402

from fetch_history import (HIST_DIR, RAIN_POINTS, RAW_DIR,  # noqa: E402
                           load_state, save_state)

FC_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
PAUSE_S = 0.4
VARS = "precipitation_previous_day1,precipitation_previous_day2"

# (year, start, end) partitions; 2024 starts at coverage, 2026 is partial
FC_WINDOWS = [(2024, "2024-06-01", "2024-12-31"),
              (2025, "2025-01-01", "2025-12-31"),
              (2026, "2026-01-01", "2026-08-05")]

SRC_FC = ("Open-Meteo Previous Runs API, archived model precipitation "
          "forecasts issued ~1/2 days ahead (best_match; retrieved "
          "historically - these are what the model actually predicted "
          "at the time, NOT observations)")


def partition_ok(pid: str, year: int, n_expected: int) -> bool:
    pq = HIST_DIR / "rainfall_fc" / pid / f"{year}.parquet"
    sc = HIST_DIR / "rainfall_fc" / pid / f"{year}.provenance.json"
    if not (pq.exists() and sc.exists()):
        return False
    try:
        side = json.load(open(sc, encoding="utf-8"))
        return side.get("rows", 0) >= n_expected * 0.98
    except (ValueError, OSError):
        return False


def main() -> int:
    state = load_state()
    fc_state = state.setdefault("rainfall_fc", {})
    n_done = n_fetched = n_failed = 0

    for p in RAIN_POINTS:
        st = fc_state.setdefault(p["id"], {})
        for year, start, end in FC_WINDOWS:
            n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            n_exp = n_days * 24
            if st.get(str(year)) == "done" and partition_ok(p["id"], year, n_exp):
                n_done += 1
                continue
            retrieved_at = utc_now_iso()
            params = {"latitude": p["lat"], "longitude": p["lon"],
                      "start_date": start, "end_date": end,
                      "hourly": VARS, "timezone": "UTC"}
            try:
                api = fetch_json(FC_API, params)
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                with gzip.open(RAW_DIR / f"rainfall_fc_{p['id']}_{year}.json.gz",
                               "wt", encoding="utf-8") as fh:
                    json.dump(api, fh)
                df = pd.DataFrame({
                    "time": api["hourly"]["time"],
                    "fc_prev1_mm": api["hourly"]["precipitation_previous_day1"],
                    "fc_prev2_mm": api["hourly"]["precipitation_previous_day2"],
                })
                out_dir = HIST_DIR / "rainfall_fc" / p["id"]
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_dir / f"{year}.parquet", index=False)
                nn = int(df["fc_prev1_mm"].notna().sum())
                save_json(out_dir / f"{year}.provenance.json", {
                    "point": p["id"], "year": year, "rows": int(len(df)),
                    "non_null_prev1": nn,
                    "columns": list(df.columns),
                    "class": "FORECAST", "archived": True,
                    "source": SRC_FC, "retrieved_at": retrieved_at,
                    "window": [start, end],
                    "partial_test_only": year == 2026,
                    "note": ("archived issued forecasts; class FORECAST "
                             "because these are model predictions, retrieved "
                             "after the fact"
                             + ("; 2026 partial, test-only" if year == 2026 else "")),
                })
                st[str(year)] = "done"
                n_fetched += 1
                print(f"  fc {p['id']:12s} {year} rows={len(df)} non_null={nn}")
            except RuntimeError as err:
                save_failed_request(f"rainfall_fc_{p['id']}_{year}", FC_API,
                                    params, str(err))
                st[str(year)] = "pending"
                n_failed += 1
                print(f"  fc {p['id']:12s} {year} FAILED: {err}")
            save_state(state)
            time.sleep(PAUSE_S)

    pending = [(pid, y) for pid, ys in fc_state.items()
               for y, s in ys.items() if s != "done"]
    print(f"FC backfill: {n_done} already done, {n_fetched} fetched, "
          f"{n_failed} failed, {len(pending)} pending")
    if pending:
        print("PENDING (rerun to resume):", pending[:10])
        return 1
    print(f"OK fc backfill complete: {len(RAIN_POINTS)} points x "
          f"{len(FC_WINDOWS)} partitions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
