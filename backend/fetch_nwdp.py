"""NWDP (National Water Data Portal, nwdp.nwic.gov.in) open-data ingest.

Downloads published CSV dataset resources for the north-east (observed
hourly rain gauges for Assam / Arunachal Pradesh / Meghalaya, observed
river levels for Assam's state telemetry, and CWC's observed river
discharge for Arunachal Pradesh) and archives them as per-station parquet
with provenance sidecars under data/history/nwdp/.

Rules: these are DATASET DOWNLOADS from an
official open-data portal built for that purpose - not portal scraping.
One-time snapshots; files covering the current window (*_2026_2030) are
re-downloaded only with --refresh or when older than REFRESH_DAYS.
Every value is OBSERVED gauge data, attributed to NWIC + the producing
agency. Bad station coordinates are flagged, never silently fixed.
"""
from __future__ import annotations
import re
import sys
import time
import datetime
import json
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
NWDP_DIR = BASE_DIR / "data" / "history" / "nwdp"
RAW_DIR = NWDP_DIR / "raw"
REFRESH_DAYS = 7

# NE bounding box for coordinate sanity (some NWDP station metadata is wrong)
NE_BBOX = (21.0, 88.0, 30.5, 98.5)  # lat_min, lon_min, lat_max, lon_max

PORTAL = "https://nwdp.nwic.gov.in"
R = [
    # (org, kind, dataset title, filename, resource path)
    ("assam", "river_level", "River Water Level (Telemetry - Hourly), Assam Department",
     "rwl_tel_hr_assam_999_2021_2025.csv",
     "/dataset/6273c426-32f9-4fdf-b67f-e4e7a46d8554/resource/51640870-5961-4696-b986-b744231f1c9f/download/rwl_tel_hr_assam_999_2021_2025.csv"),
    ("assam", "river_level", "River Water Level (Telemetry - Hourly), Assam Department",
     "rwl_tel_hr_assam_999_2026_2030.csv",
     "/dataset/6273c426-32f9-4fdf-b67f-e4e7a46d8554/resource/847f5630-f231-46c0-922d-0f2f379a5cb8/download/rwl_tel_hr_assam_999_2026_2030.csv"),
    ("assam", "rainfall", "Rainfall (Telemetry Hourly), Assam Water Department",
     "rainfall_tel_hr_assam_as_1991_2020.csv",
     "/dataset/7c8adfd6-2f7e-4bcc-b23d-f22a815e4b3c/resource/b27140e2-7c35-4d58-85b8-b2a967c3a001/download/rainfall_tel_hr_assam_as_1991_2020.csv"),
    ("assam", "rainfall", "Rainfall (Telemetry Hourly), Assam Water Department",
     "rainfall_tel_hr_assam_as_2021_2025.csv",
     "/dataset/7c8adfd6-2f7e-4bcc-b23d-f22a815e4b3c/resource/ed78ae27-56c8-4f12-a843-0654ee3a696b/download/rainfall_tel_hr_assam_as_2021_2025.csv"),
    ("assam", "rainfall", "Rainfall (Telemetry Hourly), Assam Water Department",
     "rainfall_tel_hr_assam_as_2026_2030.csv",
     "/dataset/7c8adfd6-2f7e-4bcc-b23d-f22a815e4b3c/resource/e72d7889-0ab9-494f-8e6c-e5a85b71c9fb/download/rainfall_tel_hr_assam_as_2026_2030.csv"),
    ("arunachal", "rainfall", "Rainfall (Telemetry - Hourly), Arunachal Pradesh SW Department",
     "rainfall_tel_hr_arunachal_sw_ar_2021_2025.csv",
     "/dataset/0ee54709-d8e2-4c93-8db6-044d3f0d2cd7/resource/d2ef4178-b42d-4469-8c1c-048ff4f4181f/download/rainfall_tel_hr_arunachal_sw_ar_2021_2025.csv"),
    ("arunachal", "rainfall", "Rainfall (Telemetry - Hourly), Arunachal Pradesh SW Department",
     "rainfall_tel_hr_arunachal_sw_ar_2026_2030.csv",
     "/dataset/0ee54709-d8e2-4c93-8db6-044d3f0d2cd7/resource/126d6c42-0373-4961-8009-0aa27d7e3969/download/rainfall_tel_hr_arunachal_sw_ar_2026_2030.csv"),
    ("meghalaya", "rainfall", "Rainfall (Telemetry - Hourly), Meghalaya Department",
     "rainfall_tel_hr_meghalaya_ml_2021_2025.csv",
     "/dataset/b2871e52-aa81-44a3-94ac-613859d3b401/resource/d7db8c8c-c348-4897-ab27-c7fefc6f603e/download/rainfall_tel_hr_meghalaya_ml_2021_2025.csv"),
    ("meghalaya", "rainfall", "Rainfall (Telemetry - Hourly), Meghalaya Department",
     "rainfall_tel_hr_meghalaya_ml_2026_2030.csv",
     "/dataset/b2871e52-aa81-44a3-94ac-613859d3b401/resource/1efb7056-4f70-454d-bb35-d02ec39ec6a7/download/rainfall_tel_hr_meghalaya_ml_2026_2030.csv"),
    ("manipur", "river_level", "River Water Level (Telemetry - Hourly), Manipur Surface Water Department",
     "rwl_tel_hr_manipur_999_2021_2025.csv",
     "/dataset/52ad5caf-4262-49a6-b1b7-d243164f5b75/resource/17096cf3-8e7d-43d1-a1d9-6b9871758cf6/download/rwl_tel_hr_manipur_999_2021_2025.csv"),
    ("manipur", "river_level", "River Water Level (Telemetry - Hourly), Manipur Surface Water Department",
     "rwl_tel_hr_manipur_999_2026_2030.csv",
     "/dataset/52ad5caf-4262-49a6-b1b7-d243164f5b75/resource/0dc50e29-08d1-4e5c-a8fb-d72777cad2f3/download/rwl_tel_hr_manipur_999_2026_2030.csv"),
    ("cwc_arunachal", "discharge", "River Discharge (Telemetry - Hourly), Central Water Commission (CWC)",
     "river_discharge_tele_hr_cwc_ar_1970_2025.csv",
     "/dataset/aee818d7-2cb6-4790-aa3c-126a72621170/resource/9c755c40-389e-4f5d-92c6-b936d28e51b3/download/river_discharge_tele_hr_cwc_ar_1970_2025.csv"),
    ("cwc_arunachal", "discharge", "River Discharge (Telemetry - Hourly), Central Water Commission (CWC)",
     "river_discharge_tele_hr_cwc_ar_2026_2030.csv",
     "/dataset/aee818d7-2cb6-4790-aa3c-126a72621170/resource/9a7feb81-191e-45df-b808-cd9523a97356/download/river_discharge_tele_hr_cwc_ar_2026_2030.csv"),
]

UNITS = {"rainfall": "mm/h", "river_level": "m", "discharge": "m3/s"}


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    return s[:60] or "station"


def need_download(path: Path, refresh: bool) -> bool:
    if not path.exists():
        return True
    if "_2026_2030" in path.name:
        if refresh:
            return True
        age_d = (time.time() - path.stat().st_mtime) / 86400
        return age_d > REFRESH_DAYS
    return False


def download(url: str, dest: Path) -> None:
    """Download and store gzip-compressed (dest ends .csv.gz)."""
    import gzip
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(dest)


def main() -> int:
    refresh = "--refresh" in sys.argv
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()
    n_dl = n_kept = n_fail = 0
    # group raw files per (org, kind) then split per station
    groups: dict[tuple, list[Path]] = {}
    for org, kind, title, fname, path in R:
        dest = RAW_DIR / (fname + ".gz")
        if need_download(dest, refresh):
            try:
                download(PORTAL + path, dest)
                n_dl += 1
                print(f"  downloaded {fname} ({dest.stat().st_size/1e6:.1f} MB)")
            except Exception as err:  # noqa: BLE001 - degrade honestly, never fabricate
                n_fail += 1
                print(f"  FAILED {fname}: {err}")
                (NWDP_DIR / "FAILED_DOWNLOADS.json").write_text(json.dumps(
                    {"file": fname, "error": str(err), "at": now}), encoding="utf-8")
                continue
        else:
            n_kept += 1
        groups.setdefault((org, kind, title), []).append(dest)

    total_stations = 0
    for (org, kind, title), files in groups.items():
        frames = []
        for f in files:
            try:
                frames.append(pd.read_csv(f, low_memory=False))
            except Exception as err:  # noqa: BLE001
                print(f"  PARSE FAILED {f.name}: {err}")
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        vcol = df.columns[-1]
        df["t"] = pd.to_datetime(df["Data Acquisition Time"],
                                 format="%d-%m-%Y %H:%M", errors="coerce")
        df = df.dropna(subset=["t"])
        out_root = NWDP_DIR / f"{org}_{kind}"
        out_root.mkdir(parents=True, exist_ok=True)
        for st, grp in df.groupby("Station"):
            sdir = out_root / slug(str(st))
            sdir.mkdir(exist_ok=True)
            lat = float(grp["Latitude"].iloc[0]); lon = float(grp["Longitude"].iloc[0])
            coord_ok = (NE_BBOX[0] <= lat <= NE_BBOX[2]) and (NE_BBOX[1] <= lon <= NE_BBOX[3])
            keep = grp[["t", vcol]].rename(columns={vcol: "value"}).sort_values("t")
            keep = keep.dropna(subset=["value"])
            keep.to_parquet(sdir / "data.parquet", index=False)
            prov = {
                "source": f"NWIC National Water Data Portal (nwdp.nwic.gov.in) - {title}",
                "retrieved_at": now, "class": "OBSERVED", "archived": True,
                "station": str(st), "district": str(grp["District"].iloc[0]),
                "lat": lat, "lon": lon, "unit": UNITS[kind],
                "rows": int(len(keep)),
                "first_utc": str(keep["t"].min()), "last_utc": str(keep["t"].max()),
                "note": ("station coordinates fall OUTSIDE the north-east bounding box "
                         "- portal metadata error, position untrusted") if not coord_ok else "",
            }
            (sdir / "data.provenance.json").write_text(
                json.dumps(prov, indent=1), encoding="utf-8")
            total_stations += 1
        print(f"  {org}_{kind}: {df['Station'].nunique()} stations, {len(df)} rows")

    print(f"OK nwdp: {n_dl} downloaded, {n_kept} kept, {n_fail} failed; "
          f"{total_stations} station archives -> data/history/nwdp/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
