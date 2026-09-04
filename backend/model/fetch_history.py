"""MODEL v0, STEP M1 (generalised in Phase 1) - historical backfill
2015-2025 (+2026 partial) per basin, and discharge-only history for the
remaining PoC river cells.

    python backend/model/fetch_history.py                    # basin dikhow (default)
    python backend/model/fetch_history.py --basin kopili
    python backend/model/fetch_history.py --basin jiabharali
    python backend/model/fetch_history.py --all-cells        # discharge-only, other 5 cells

Sources (free, no key):
- Open-Meteo Archive API: hourly precipitation, ERA5-family reanalysis, for
  the basin's catchment anchor cells (backend/model/basins.py, CATCHMENT.md).
- Open-Meteo Flood API (GloFAS v4 reanalysis, DAILY) with start/end dates
  for the target cell + probed upstream candidates.

HONESTY: discharge is GloFAS reanalysis - a MODELLED product, not observed
river data; observed CWC gauge data will replace it when access is granted.
All values class OBSERVED with exact product names in source strings.

Storage (idempotent, resumable, rate-limit-friendly):
- data/history/rainfall/<point>/<year>.parquet   (+ <year>.provenance.json)
- data/history/discharge/<point>/<year>.parquet  (+ sidecar)
- data/history/raw/<kind>_<point>_<year>.json.gz (raw API response, cached)
- BACKFILL-STATE.json (repo root): per kind/point/year done|pending; an
  interrupted run resumes without refetching anything marked done whose
  files still exist and pass a row-count sanity check.
- One request per point-year, ~0.4 s pause, exponential backoff inside
  fetch_json. 2026 is fetched Jan 1 - Aug 5 only and flagged
  partial/test_only (never used in training).

Also rewrites data/history/GAPS.md from what is on disk - per-partition row
counts vs expected and missing spans, monsoon-month (May-Oct) gaps flagged.

Exit codes: 0 = backfill complete (all done); 1 = pending partitions remain
(rerun to resume).
"""

import datetime
import gzip
import json
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import (DATA_DIR, REPO_ROOT, fetch_json, load_json, save_json,  # noqa: E402
                    utc_now_iso)

from basins import BASINS, DISCHARGE_ONLY_CELLS  # noqa: E402

HIST_DIR = DATA_DIR / "history"
RAW_DIR = HIST_DIR / "raw"
STATE_PATH = REPO_ROOT / "BACKFILL-STATE.json"

YEARS = list(range(2015, 2026))          # full training/validation years
PARTIAL_YEAR = 2026                       # test-only, Jan 1 - Aug 5
PARTIAL_END = "2026-08-05"

RAIN_API = "https://archive-api.open-meteo.com/v1/archive"
FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
PAUSE_S = 0.4

SRC_RAIN = ("Open-Meteo Historical Weather (Archive) API, ERA5-family "
            "reanalysis (model analysis of past weather, NOT gauge data)")
SRC_DISCH = ("GloFAS v4 reanalysis via Open-Meteo Flood API (MODELLED "
             "product, not observed river data; observed CWC gauge data "
             "will replace this when access is granted)")

# backward-compatible module constants (Dikhow) used by predict.py etc.
RAIN_POINTS = BASINS["dikhow"]["rain_points"]
MON_POINT = RAIN_POINTS[-1]


def load_state() -> dict:
    if STATE_PATH.exists():
        st = load_json(STATE_PATH)
        # migrate the pre-basin layout
        if "river_points" in st and "river_points_by_basin" not in st:
            st["river_points_by_basin"] = {"dikhow": st.pop("river_points")}
        st.setdefault("river_points_by_basin", {})
        return st
    return {"created_at": utc_now_iso(), "rainfall": {}, "discharge": {},
            "river_points_by_basin": {}, "note": "per point+year backfill state; "
            "done partitions are never refetched"}


def save_state(state: dict) -> None:
    state["updated_at"] = utc_now_iso()
    save_json(STATE_PATH, state)


def year_window(year: int) -> tuple[str, str, bool]:
    if year == PARTIAL_YEAR:
        return f"{year}-01-01", PARTIAL_END, True
    return f"{year}-01-01", f"{year}-12-31", False


def expected_rows(year: int, hourly: bool) -> int:
    start, end, _ = year_window(year)
    days = (datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)).days + 1
    return days * 24 if hourly else days


def partition_ok(kind: str, pid: str, year: int) -> bool:
    pq = HIST_DIR / kind / pid / f"{year}.parquet"
    sc = HIST_DIR / kind / pid / f"{year}.provenance.json"
    if not (pq.exists() and sc.exists()):
        return False
    try:
        return load_json(sc).get("rows", 0) >= expected_rows(year, kind == "rainfall") * 0.98
    except (ValueError, OSError):
        return False


def save_partition(kind: str, pid: str, year: int, df: pd.DataFrame,
                   source: str, retrieved_at: str, partial: bool, basin: str) -> None:
    out_dir = HIST_DIR / kind / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{year}.parquet", index=False)
    save_json(out_dir / f"{year}.provenance.json", {
        "point": pid, "year": year, "rows": int(len(df)), "basin": basin,
        "columns": list(df.columns),
        "class": "OBSERVED", "source": source, "retrieved_at": retrieved_at,
        "window": list(year_window(year)[:2]),
        "partial_test_only": partial,
        "note": ("PARTIAL year, test-only, never used in training" if partial else "full year"),
    })


def save_raw(kind: str, pid: str, year: int, api: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(RAW_DIR / f"{kind}_{pid}_{year}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(api, fh)


def fetch_rain_year(p: dict, year: int) -> pd.DataFrame:
    start, end, _ = year_window(year)
    api = fetch_json(RAIN_API, {"latitude": p["lat"], "longitude": p["lon"],
                                "start_date": start, "end_date": end,
                                "hourly": "precipitation", "timezone": "UTC"})
    save_raw("rainfall", p["id"], year, api)
    return pd.DataFrame({"time": api["hourly"]["time"],
                         "precipitation_mm": api["hourly"]["precipitation"]})


def fetch_disch_year(p: dict, year: int) -> pd.DataFrame:
    start, end, _ = year_window(year)
    api = fetch_json(FLOOD_API, {"latitude": p["lat"], "longitude": p["lon"],
                                 "start_date": start, "end_date": end,
                                 "daily": "river_discharge", "timezone": "UTC"})
    save_raw("discharge", p["id"], year, api)
    return pd.DataFrame({"date": api["daily"]["time"],
                         "discharge_m3s": api["daily"]["river_discharge"]})


def live_cells() -> dict:
    """Snapped GloFAS cells of the PoC river points from data/discharge.json."""
    try:
        live = load_json(DATA_DIR / "discharge.json")
        return {p["id"]: (p["grid_lat"], p["grid_lon"]) for p in live["points"]}
    except (FileNotFoundError, ValueError, KeyError):
        return {}


def river_points_for(basin: str, state: dict) -> list[dict]:
    """Target cell + resolved upstream cells for a basin; cached in state."""
    cached = state["river_points_by_basin"].get(basin)
    if cached:
        return cached
    cfg = BASINS[basin]
    cells = live_cells()
    if cfg["target"] not in cells:
        raise SystemExit(f"{cfg['target']} not in data/discharge.json - run the live pipeline first")
    lat, lon = cells[cfg["target"]]
    points = [{"id": cfg["target"], "lat": lat, "lon": lon, "role": "target"}]
    from fetch_discharge import snap_to_river_cell
    for cand in cfg["upstream"]:
        if cand.get("existing"):
            # a cell that already has history (PoC target or another basin's
            # cell) used as an upstream FEATURE - no probe, no refetch
            if cand["id"] in cells:
                lat, lon = cells[cand["id"]]
            else:
                hist_dir = HIST_DIR / "discharge" / cand["id"]
                if not hist_dir.exists():
                    print(f"  upstream {cand['id']}: no history and not live - dropped")
                    continue
                lat = lon = None
            points.append({"id": cand["id"], "lat": lat, "lon": lon, "role": "upstream-existing"})
            print(f"  upstream {cand['id']}: existing cell reused (no probe)")
            continue
        try:
            clat, clon, mean = snap_to_river_cell(cand)
        except RuntimeError as err:
            print(f"  probe {cand['id']}: FAILED ({err}) - dropped")
            continue
        dup = any(abs(clat - q["lat"]) < 0.001 and abs(clon - q["lon"]) < 0.001 for q in points)
        qmax = cand.get("qmax")  # None = no upper bound (huge mainstem cells)
        plaus = (mean is not None and cand["qmin"] <= mean
                 and (qmax is None or mean <= qmax))
        if dup or not plaus:
            print(f"  probe {cand['id']}: cell ({clat},{clon}) mean={mean} - "
                  f"{'duplicate' if dup else 'implausible'} - dropped")
            continue
        print(f"  probe {cand['id']}: cell ({clat},{clon}) 7-day mean {mean:.0f} m3/s - kept")
        points.append({"id": cand["id"], "lat": clat, "lon": clon, "role": "upstream"})
        time.sleep(PAUSE_S)
    state["river_points_by_basin"][basin] = points
    save_state(state)
    return points


def run_jobs(jobs: list, state: dict, basin: str) -> tuple[int, int, int]:
    n_done = n_fetched = n_failed = 0
    for kind, p, year in jobs:
        st = state[kind].setdefault(p["id"], {})
        if st.get(str(year)) == "done" and partition_ok(kind, p["id"], year):
            n_done += 1
            continue
        retrieved_at = utc_now_iso()
        _, _, partial = year_window(year)
        try:
            df = fetch_rain_year(p, year) if kind == "rainfall" else fetch_disch_year(p, year)
            save_partition(kind, p["id"], year, df,
                           SRC_RAIN if kind == "rainfall" else SRC_DISCH,
                           retrieved_at, partial, basin)
            st[str(year)] = "done"
            n_fetched += 1
            print(f"  {kind:9s} {p['id']:22s} {year} rows={len(df)}"
                  f"{' (partial, test-only)' if partial else ''}")
        except RuntimeError as err:
            from common import save_failed_request
            save_failed_request(f"history_{kind}_{p['id']}_{year}",
                                RAIN_API if kind == "rainfall" else FLOOD_API, {}, str(err))
            st[str(year)] = "pending"
            n_failed += 1
            print(f"  {kind:9s} {p['id']:22s} {year} FAILED: {err}")
        save_state(state)
        time.sleep(PAUSE_S)
    return n_done, n_fetched, n_failed


def gaps_report() -> None:
    """Disk scan of every partition (all basins) -> data/history/GAPS.md."""
    lines = ["# GAPS.md - backfill completeness report", "",
             f"Generated {utc_now_iso()} by backend/model/fetch_history.py from the",
             "partitions on disk (all basins). Expected rows = hours (rainfall) or",
             "days (discharge) in the partition window. Monsoon = May-October.", ""]
    loud = 0
    for kind, hourly in (("rainfall", True), ("discharge", False)):
        lines += [f"## {kind}", "", "| point | year | rows | expected | nulls | monsoon gaps |",
                  "|---|---|---|---|---|---|"]
        base = HIST_DIR / kind
        for pdir in sorted(base.glob("*")) if base.exists() else []:
            for year in YEARS + [PARTIAL_YEAR]:
                pq = pdir / f"{year}.parquet"
                if not pq.exists():
                    lines.append(f"| {pdir.name} | {year} | MISSING | {expected_rows(year, hourly)} | - | **PARTITION MISSING** |")
                    loud += 1
                    continue
                df = pd.read_parquet(pq)
                col, tcol = ("precipitation_mm", "time") if hourly else ("discharge_m3s", "date")
                ts = pd.to_datetime(df[tcol])
                nulls = int(df[col].isna().sum())
                mn = int(df[col][(ts.dt.month >= 5) & (ts.dt.month <= 10)].isna().sum())
                flag = ""
                if mn:
                    flag = f"**{mn} null values in monsoon months**"
                    loud += 1
                exp = expected_rows(year, hourly)
                if len(df) < exp:
                    flag = (flag + " " if flag else "") + f"**{exp - len(df)} rows short**"
                    loud += 1
                lines.append(f"| {pdir.name} | {year} | {len(df)} | {exp} | {nulls} | {flag or 'none'} |")
        lines.append("")
    lines.append(f"**Loud flags: {loud}**" if loud else "**No monsoon gaps, no missing partitions.**")
    lines += ["", "Note: GloFAS discharge is DAILY only; 2026 partitions are partial by",
              "design (Jan 1 - Aug 5, test-only, never trained on)."]
    (HIST_DIR / "GAPS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  GAPS.md written ({'FLAGS: ' + str(loud) if loud else 'clean'})")


def main() -> int:
    args = sys.argv[1:]
    basin = args[args.index("--basin") + 1] if "--basin" in args else "dikhow"
    all_cells = "--all-cells" in args
    state = load_state()
    jobs = []
    if all_cells:
        cells = live_cells()
        for cid in DISCHARGE_ONLY_CELLS:
            if cid not in cells:
                print(f"  {cid}: not in data/discharge.json - skipped")
                continue
            lat, lon = cells[cid]
            for year in YEARS + [PARTIAL_YEAR]:
                jobs.append(("discharge", {"id": cid, "lat": lat, "lon": lon}, year))
        label = "discharge-only cells"
    else:
        if basin not in BASINS:
            raise SystemExit(f"unknown basin {basin!r}; known: {list(BASINS)}")
        print(f"Basin {basin}: probing upstream GloFAS cells...")
        river_points = river_points_for(basin, state)
        for p in BASINS[basin]["rain_points"]:
            for year in YEARS + [PARTIAL_YEAR]:
                jobs.append(("rainfall", p, year))
        for p in river_points:
            for year in YEARS + [PARTIAL_YEAR]:
                jobs.append(("discharge", p, year))
        label = f"basin {basin}"

    n_done, n_fetched, n_failed = run_jobs(jobs, state, "all-cells" if all_cells else basin)
    gaps_report()

    pending = [(k, pid, y) for k in ("rainfall", "discharge")
               for pid, ys in state[k].items() for y, s in ys.items() if s != "done"]
    print(f"Backfill {label}: {n_done} already done, {n_fetched} fetched, "
          f"{n_failed} failed, {len(pending)} pending (all basins)")
    if pending:
        print("PENDING (rerun to resume):", pending[:10])
        return 1
    print(f"OK backfill complete for {label}; state -> {STATE_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
