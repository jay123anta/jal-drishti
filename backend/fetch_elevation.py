"""Village exposure layer: terrain elevation + distance to the nearest river.

A village's flood exposure depends on real, static terrain: how low it
sits and how close it is to a river. This fetches ground elevation for
every village and river point (Open-Meteo elevation API, a Copernicus
DEM, free, no key) and writes data/elevation.json.

classify_risk attaches an "exposure" note to each village from this.
IMPORTANT: exposure is terrain CONTEXT (class OBSERVED, a DEM fact) shown
beside the colour - it is never mixed into the risk colour, because we
have no validated village-level flood outcomes to justify that. Elevation
is static, so this is fetched once and reused (--refresh to redo).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))
from common import DATA_DIR, load_json  # noqa: E402

OUT = DATA_DIR / "elevation.json"
API = "https://api.open-meteo.com/v1/elevation"
SRC = "Open-Meteo elevation API (Copernicus GLO-90 digital elevation model)"


def fetch_batch(coords: list[tuple]) -> list[float]:
    lats = ",".join(f"{a:.5f}" for a, _ in coords)
    lons = ",".join(f"{b:.5f}" for _, b in coords)
    r = requests.get(API, params={"latitude": lats, "longitude": lons}, timeout=60)
    r.raise_for_status()
    return r.json()["elevation"]


def main() -> int:
    villages = load_json(DATA_DIR / "villages.json")["villages"]
    try:
        rivers = load_json(DATA_DIR / "discharge.json")["points"]
    except (FileNotFoundError, ValueError):
        rivers = []

    points = [("v", v["name"], v["district"], v["lat"], v["lon"]) for v in villages]
    points += [("r", p["id"], "", p.get("lat"), p.get("lon")) for p in rivers
               if p.get("lat") is not None]

    existing = {}
    if OUT.exists() and "--refresh" not in sys.argv:
        try:
            existing = {e["key"]: e for e in load_json(OUT)["points"]}
        except (ValueError, KeyError):
            existing = {}

    def key(kind, a, b):
        return f"{kind}:{a}|{b}"

    todo = [p for p in points if key(p[0], p[1], p[2]) not in existing]
    print(f"elevation: {len(points)} points, {len(existing)} cached, {len(todo)} to fetch")

    out = list(existing.values())
    for i in range(0, len(todo), 90):
        batch = todo[i:i + 90]
        try:
            elevs = fetch_batch([(p[3], p[4]) for p in batch])
        except Exception as err:  # noqa: BLE001 - degrade honestly
            print(f"  batch {i} failed ({err}) - kept what we had")
            break
        for p, e in zip(batch, elevs):
            out.append({"key": key(p[0], p[1], p[2]), "kind": p[0], "name": p[1],
                        "district": p[2], "lat": p[3], "lon": p[4], "elevation_m": e})
        time.sleep(1)

    OUT.write_text(json.dumps({
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": SRC, "class": "OBSERVED", "unit": "m",
        "points": out}, indent=1), encoding="utf-8")
    print(f"OK elevation: {len(out)} points -> data/elevation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
