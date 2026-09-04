"""STRETCH (documented, NOT run by the pipeline) - Sentinel-1 water extent
for the June 2022 Barak flood.

This script is the honest boundary of the PoC: scene FOOTPRINTS and
quicklooks are public (Step 5 ships them), but PIXEL data needs a free
Copernicus Data Space Ecosystem (CDSE) account. Rather than fabricate a
water mask, this file contains the exact, ready-to-run steps.

What it does when run:
1. No credentials  -> prints the full recipe below and exits 2.
2. CDSE_USERNAME / CDSE_PASSWORD set -> obtains an access token and
   downloads the best pre-flood / peak-flood GRD pair (same relative
   orbit) selected from public/s1_footprints.geojson, then prints the
   SNAP processing commands. It still does NOT invent a water mask.

The recipe (also printed at runtime):
  a. Register (free): https://dataspace.copernicus.eu
  b. export CDSE_USERNAME=... CDSE_PASSWORD=...   (or set in PowerShell)
  c. python backend/s1_water_extent_stretch.py    -> downloads the 2 .zip GRDs
  d. Process each with ESA SNAP's gpt (free download, includes graphs):
       gpt Calibration -Ssource=<scene>.zip -PoutputSigmaBand=true -t cal.dim
       gpt Terrain-Correction -Ssource=cal.dim -PdemName="SRTM 3Sec" -t tc.dim
  e. Water mask: threshold Sigma0_VV < 0.0025 (~ -26 dB .. tune -15..-26 dB
     against the quicklooks); flood extent = water(peak) AND NOT water(pre).
  f. Any raster produced this way would be class OBSERVED (derived), source
     "Sentinel-1 GRD via CDSE, SNAP Cal+TC, VV threshold" - label the
     threshold choice SIMULATED until validated against ground truth.
"""

import os
import sys

import requests

from common import PUBLIC_DIR, load_json

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")


def pick_pair(features: list[dict]) -> tuple[dict, dict]:
    """Best pre/peak pair: prefer same orbit_state so geometry matches."""
    pre = [f["properties"] for f in features if f["properties"]["window"] == "pre_flood"]
    peak = [f["properties"] for f in features if f["properties"]["window"] == "peak_flood"]
    if not pre or not peak:
        raise SystemExit("s1_footprints.geojson lacks one of the two windows - rerun Step 5")
    for pk in peak:
        same = [p for p in pre if p["orbit_state"] == pk["orbit_state"]]
        if same:
            # latest matching pre-flood scene = shortest revisit gap
            return max(same, key=lambda p: p["acquired"]), pk
    return max(pre, key=lambda p: p["acquired"]), peak[0]


def main() -> int:
    doc = load_json(PUBLIC_DIR / "s1_footprints.geojson")
    pre, peak = pick_pair(doc["features"])
    print("Selected pair (same orbit state where possible):")
    for tag, p in (("PRE ", pre), ("PEAK", peak)):
        print(f"  {tag} {p['scene_id']}  acquired={p['acquired']}  orbit={p['orbit_state']}")

    user, pwd = os.environ.get("CDSE_USERNAME"), os.environ.get("CDSE_PASSWORD")
    if not (user and pwd):
        print("\nNo CDSE_USERNAME / CDSE_PASSWORD in the environment.")
        print("Pixel download needs a free account - full recipe in this file's docstring.")
        print("Nothing was fabricated: the PoC ships footprints + quicklooks only.")
        return 2

    tok = requests.post(TOKEN_URL, data={
        "grant_type": "password", "client_id": "cdse-public",
        "username": user, "password": pwd,
    }, timeout=60)
    tok.raise_for_status()
    headers = {"Authorization": f"Bearer {tok.json()['access_token']}"}

    for p in (pre, peak):
        out = PUBLIC_DIR.parent / "data" / "raw" / f"{p['scene_id']}.zip"
        if out.exists():
            print(f"  already downloaded: {out.name}")
            continue
        print(f"  downloading {p['scene_id']} (~1 GB)...")
        with requests.get(p["product_download_href"], headers=headers,
                          stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        print(f"  saved {out}")

    print("\nNext: run the SNAP gpt commands from the docstring (steps d-f).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
