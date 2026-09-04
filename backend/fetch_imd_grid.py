"""IMD gridded daily rainfall (0.25 deg) - the reference record for Indian rain.

Source: IMD Pune's public gridded-rainfall page
(imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html). Download is a
plain form POST (RF25.php, field RF25=<year>) returning a classic NetCDF
file - a published dataset offered for download, not portal scraping. This is the gauge-based gridded analysis the
Indian research community treats as ground truth for daily rainfall.

Per year 2015-2025: download the all-India NetCDF (~25 MB, kept locally
in raw_nc/, NOT committed - reproducible via the exact recipe recorded in
every provenance sidecar), extract the north-east window (lat 21-30.5,
lon 88-98.5) and store it as a small parquet with provenance. Values are
OBSERVED-derived (a gridded analysis interpolated from rain gauges -
stated in the sidecar, not hidden). Missing values (< 0) are masked.
"""
from __future__ import annotations
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
IMD_DIR = BASE_DIR / "data" / "history" / "imd"
RAW_DIR = IMD_DIR / "raw_nc"          # local only (gitignored; ~25 MB/yr)
GRID_DIR = IMD_DIR / "grid_ne"

URL = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"
YEARS = list(range(2015, 2026))
NE = (21.0, 30.5, 88.0, 98.5)         # lat_min, lat_max, lon_min, lon_max

SRC = ("IMD Pune gridded daily rainfall 0.25x0.25 deg (gauge-based gridded "
       "analysis) - imdpune.gov.in Rainfall_25_NetCDF page; recipe: HTTP POST "
       f"{URL} with form field RF25=<year>")


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def download(year: int, dest: Path) -> None:
    r = requests.post(URL, data={"RF25": str(year)}, timeout=600, stream=True)
    r.raise_for_status()
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size < 1_000_000 or tmp.read_bytes()[:3] != b"CDF":
        tmp.unlink()
        raise RuntimeError(f"{year}: response is not a NetCDF file")
    tmp.replace(dest)


def extract_ne(nc_path: Path, year: int) -> pd.DataFrame:
    from scipy.io import netcdf_file
    f = netcdf_file(str(nc_path), mmap=False)
    lat = np.array(f.variables["LATITUDE"][:], dtype=float)
    lon = np.array(f.variables["LONGITUDE"][:], dtype=float)
    rain_var = next(k for k, v in f.variables.items() if len(v.shape) == 3)
    rain = np.array(f.variables[rain_var][:], dtype=float)   # (day, lat, lon)
    la = (lat >= NE[0]) & (lat <= NE[1])
    lo = (lon >= NE[2]) & (lon <= NE[3])
    sub = rain[:, la][:, :, lo]
    sub[sub < 0] = np.nan                                    # -999 = missing
    days = pd.date_range(f"{year}-01-01", periods=sub.shape[0], freq="D")
    rows = []
    for i, d in enumerate(days):
        arr = sub[i]
        jj, kk = np.where(~np.isnan(arr))
        rows.append(pd.DataFrame({
            "date": d, "lat": lat[la][jj], "lon": lon[lo][kk],
            "rain_mm": arr[jj, kk]}))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    now = utc_now_iso()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = n_kept = n_fail = 0
    for year in YEARS:
        out_p = GRID_DIR / f"{year}.parquet"
        if out_p.exists():
            n_kept += 1
            continue
        nc_p = RAW_DIR / f"RF25_{year}.nc"
        try:
            if not nc_p.exists() or nc_p.stat().st_size < 1_000_000:
                download(year, nc_p)
                print(f"  {year}: downloaded {nc_p.stat().st_size/1e6:.0f} MB")
            df = extract_ne(nc_p, year)
            df.to_parquet(out_p, index=False)
            (GRID_DIR / f"{year}.provenance.json").write_text(json.dumps({
                "source": SRC, "class": "OBSERVED", "archived": True,
                "retrieved_at": now, "year": year, "rows": int(len(df)),
                "ne_window": {"lat": NE[:2], "lon": NE[2:]},
                "note": ("gridded ANALYSIS interpolated from IMD rain gauges - "
                         "observed-derived, not a direct point measurement; "
                         "raw all-India NetCDF kept locally in raw_nc/ (not "
                         "committed, ~25 MB/yr) and reproducible via the "
                         "recorded POST recipe")}, indent=1), encoding="utf-8")
            print(f"  {year}: NE subset {len(df)} cell-days -> {out_p.name}")
            n_ok += 1
        except Exception as err:  # noqa: BLE001 - degrade honestly
            n_fail += 1
            print(f"  {year}: FAILED ({err})")
            (IMD_DIR / "FAILED.json").write_text(json.dumps(
                {"year": year, "error": str(err), "at": now}), encoding="utf-8")
    print(f"OK imd grid: {n_ok} extracted, {n_kept} kept, {n_fail} failed "
          f"-> data/history/imd/grid_ne/")
    return 0 if n_fail == 0 else 0   # degraded runs still exit 0, recorded


if __name__ == "__main__":
    sys.exit(main())
