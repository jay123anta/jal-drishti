"""NASA GPM IMERG satellite rainfall - observed rain over the whole catchment.

IMERG sees the Bhutan and Tibet headwaters that India's gauge networks
cannot. Source: NASA PPS (arthurhouhttps.pps.eosdis.nasa.gov), the daily
GIS GeoTIFF product - tiny (~0.2 MB/day), global 0.1 deg. Login is the
registered PPS email used as BOTH username and password; read from the
IMERG_EMAIL environment variable, never stored in the repo.

Decoding (from each file's own ImageDescription tag): value is in units
of 0.1 mm/hr (a daily-mean RATE), ScaleFactor 10, so daily total mm =
raw * 0.1 * 24 = raw * 2.4; the fill value 29999 is masked. Filenames
carry a variable sequence number, so each day's directory is listed
rather than name-constructed.

Per run: for the NE window, fetch any missing days from START..yesterday
and append the per-anchor daily series to
data/history/imerg/<anchor>.parquet with provenance (class OBSERVED,
satellite-derived). Honest degradation: a day that 404s or fails to
decode is skipped and counted, never fabricated.
"""
from __future__ import annotations
import datetime
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend" / "model"))
from basins import BASINS  # noqa: E402

OUT = BASE_DIR / "data" / "history" / "imerg"
PPS = "https://arthurhouhttps.pps.eosdis.nasa.gov/gpmdata"
START = datetime.date(2015, 6, 1)
FILL = 29999
Image.MAX_IMAGE_PIXELS = None

SRC = ("NASA GPM IMERG Final daily GIS product (arthurhouhttps.pps.eosdis.nasa.gov); "
       "satellite-derived precipitation, 0.1 deg; daily mm = raw * 0.1(mm/hr) * 24")


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def anchors() -> list[dict]:
    seen, out = {}, []
    for cfg in BASINS.values():
        for pt in cfg["rain_points"]:
            if pt["id"] not in seen:
                seen[pt["id"]] = True
                out.append(pt)
    return out


def day_tif_url(auth: tuple, d: datetime.date) -> str | None:
    base = f"{PPS}/{d:%Y/%m/%d}/gis/"
    try:
        r = requests.get(base, auth=auth, timeout=60)
        if r.status_code != 200:
            return None
        m = re.search(r'3B-DAY-GIS[^"]+\.tif', r.text)
        return base + m.group(0) if m else None
    except requests.RequestException:
        return None


def sample_ne(tif_bytes: bytes, pts: list[dict]) -> dict:
    a = np.array(Image.open(io.BytesIO(tif_bytes))).astype(float)
    a[a >= FILL] = np.nan
    out = {}
    for p in pts:
        row = int(round((90 - p["lat"]) / 0.1))
        col = int(round((p["lon"] + 180) / 0.1))
        v = a[row, col]
        out[p["id"]] = None if np.isnan(v) else round(float(v) * 2.4, 1)  # -> mm/day
    return out


def load_existing() -> dict:
    data = {}
    if OUT.exists():
        for f in OUT.glob("*.parquet"):
            data[f.stem] = pd.read_parquet(f)
    return data


def main() -> int:
    email = os.environ.get("IMERG_EMAIL")
    if not email:
        print("DEGRADED: IMERG_EMAIL not set - skipped (register at "
              "registration.pps.eosdis.nasa.gov; set email as the value)")
        return 0
    auth = (email, email)
    now = utc_now_iso()
    OUT.mkdir(parents=True, exist_ok=True)
    pts = anchors()
    existing = load_existing()
    have = set()
    for df in existing.values():
        have |= set(df["date"].astype(str))
    end = datetime.date.today() - datetime.timedelta(days=1)
    days = [START + datetime.timedelta(n) for n in range((end - START).days + 1)]
    todo = [d for d in days if d.isoformat() not in have]
    print(f"IMERG: {len(pts)} anchors, {len(have)} days already archived, {len(todo)} to fetch")

    def flush(rows_by_anchor: dict) -> None:
        """Write accumulated rows to disk - safe to call repeatedly (checkpoint)."""
        for aid, rows in rows_by_anchor.items():
            if not rows:
                continue
            new = pd.DataFrame(rows)
            p = OUT / f"{aid}.parquet"
            if p.exists():
                new = pd.concat([pd.read_parquet(p), new], ignore_index=True)
            new = new.drop_duplicates(subset="date", keep="first").sort_values("date")
            new.to_parquet(p, index=False)
            (OUT / f"{aid}.provenance.json").write_text(
                f'{{"source": "{SRC}", "class": "OBSERVED", "archived": true, '
                f'"retrieved_at": "{now}", "anchor": "{aid}", "rows": {len(new)}}}',
                encoding="utf-8")

    rows_by_anchor: dict[str, list] = {}
    n_ok = n_skip = 0
    for d in todo:
        url = day_tif_url(auth, d)
        if not url:
            n_skip += 1
            continue
        try:
            r = requests.get(url, auth=auth, timeout=120)
            r.raise_for_status()
            vals = sample_ne(r.content, pts)
        except Exception:  # noqa: BLE001 - skip and count, never fabricate
            n_skip += 1
            continue
        for aid, v in vals.items():
            rows_by_anchor.setdefault(aid, []).append({"date": d.isoformat(), "rain_mm": v})
        n_ok += 1
        if n_ok % 100 == 0:
            flush(rows_by_anchor)          # checkpoint: nothing lost on interrupt
            rows_by_anchor = {}
            print(f"  {n_ok} days fetched, checkpointed...")

    flush(rows_by_anchor)                  # final partial batch
    print(f"OK imerg: {n_ok} days fetched, {n_skip} skipped -> data/history/imerg/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
