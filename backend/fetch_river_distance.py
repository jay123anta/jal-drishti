"""How far each mapped village really is from a river channel.

The terrain note in a village popup used to say "N km from the river", but the
number was the straight-line distance to the ONE GloFAS point this project
models for that basin. For a village in the Barak valley 20 km upstream of the
Silchar point, that read as "20 km from the river" when the Barak itself was a
short walk away. Distance to a measuring point is not distance to water, and a
reader takes the second meaning.

This fetches the actual river network from OpenStreetMap and computes, per
village, the distance to the nearest waterway=river / riverbank geometry.

Fetched in tiles over the village bounding box, each tile cached on disk, so a
rerun costs nothing and a partial failure resumes. OSM river geometry changes
slowly, so this is a cached static layer like elevation - it is NOT part of the
3-hourly loop and NOT part of the risk colour.

    python backend/fetch_river_distance.py            # fill gaps only
    python backend/fetch_river_distance.py --refresh  # refetch every tile

Writes data/river_distance.json (village -> km, with provenance).
Exit 0 = every village resolved; 1 = some tiles unavailable, rerun later.
"""

import json
import math
import pathlib
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import DATA_DIR, OBSERVED, load_json, save_json, utc_now_iso

CACHE = DATA_DIR / "history" / "osm_rivers"
OUT = DATA_DIR / "river_distance.json"
ENDPOINTS = ["https://overpass.private.coffee/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter",
             "https://overpass-api.de/api/interpreter",
             "https://overpass.osm.ch/api/interpreter"]
TILE = 1.0          # degrees; keeps each response small enough to survive
PAD = 0.35          # overlap so a village near a tile edge still sees its river
SRC = ("OpenStreetMap contributors via Overpass API (waterway=river and "
       "riverbank ways); straight-line distance from the village node to the "
       "nearest river geometry - terrain context, NOT part of the risk colour")


def tiles_for(pts) -> list[tuple]:
    """1-degree tiles covering every village, padded so edges overlap."""
    out = set()
    for la, lo in pts:
        out.add((math.floor(la / TILE) * TILE, math.floor(lo / TILE) * TILE))
    return sorted(out)


def fetch_tile(s_lat, s_lon):
    q = (f'[out:json][timeout:180];'
         f'way["waterway"~"^(river|riverbank)$"]'
         f'({s_lat - PAD},{s_lon - PAD},{s_lat + TILE + PAD},{s_lon + TILE + PAD});'
         f'out geom;')
    data = urllib.parse.urlencode({"data": q}).encode()
    last = None
    for i in range(8):
        ep = ENDPOINTS[i % len(ENDPOINTS)]
        try:
            req = urllib.request.Request(ep, data=data,
                                         headers={"User-Agent": "jaldrishti-river-distance"})
            with urllib.request.urlopen(req, timeout=240) as r:
                els = json.loads(r.read().decode()).get("elements", [])
            # An empty tile over Assam is not a real answer - every 1-degree tile
            # here contains rivers. Overpass returns [] on some overload paths
            # with HTTP 200, and caching that silently leaves a hole in the
            # network that shows up as villages being "far from any river".
            if not els:
                last = RuntimeError("empty element list (treated as a failed tile)")
                time.sleep(10 * (i + 1))
                continue
            return els
        except Exception as e:
            last = e
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"all endpoints failed for tile {s_lat},{s_lon}: {last}")


def main() -> int:
    refresh = "--refresh" in sys.argv
    # Read the RAW village list, not the published status file: the published
    # file is written later in the pipeline, so reading it left newly added
    # villages without a distance until the next run.
    villages = load_json(DATA_DIR / "villages.json")["villages"]
    pts = [(v["lat"], v["lon"]) for v in villages]
    CACHE.mkdir(parents=True, exist_ok=True)

    geom = []                       # flat list of (lat, lon) river vertices
    missing = []
    for s_lat, s_lon in tiles_for(pts):
        f = CACHE / f"tile_{s_lat:+06.1f}_{s_lon:+07.1f}.json"
        if f.exists() and not refresh and json.loads(f.read_text(encoding="utf-8")):
            els = json.loads(f.read_text(encoding="utf-8"))
        else:
            try:
                els = fetch_tile(s_lat, s_lon)
            except RuntimeError as e:
                print(f"  {e}")
                missing.append((s_lat, s_lon))
                continue
            f.write_text(json.dumps(els), encoding="utf-8")
            print(f"  tile {s_lat:+.1f},{s_lon:+.1f}: {len(els)} river ways cached")
            time.sleep(3)
        for el in els:
            for p in el.get("geometry") or []:
                geom.append((p["lat"], p["lon"], (el.get("tags") or {}).get("name") or ""))

    if not geom:
        print("fetch_river_distance: no river geometry available - nothing written")
        return 1

    now = utc_now_iso()
    # Vectorised: 141 villages against ~2.5M river vertices is 350M haversines in
    # pure Python (tens of minutes). numpy does the whole comparison per village
    # in one pass, after a cheap degree-box mask.
    glat = np.fromiter((g[0] for g in geom), dtype=np.float64, count=len(geom))
    glon = np.fromiter((g[1] for g in geom), dtype=np.float64, count=len(geom))
    gname = [g[2] for g in geom]
    rows = []
    for v in villages:
        la, lo = v["lat"], v["lon"]
        box = (np.abs(glat - la) <= 0.6) & (np.abs(glon - lo) <= 0.6)
        idx = np.flatnonzero(box)
        if idx.size == 0:
            rows.append({"key": f"v:{v['name']}|{v['district']}", "name": v["name"],
                         "district": v["district"], "distance_km": None,
                         "nearest_named": None})
            continue
        p1, p2 = math.radians(la), np.radians(glat[idx])
        dp = np.radians(glat[idx] - la)
        dl = np.radians(glon[idx] - lo)
        a = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        d = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        j = int(np.argmin(d))
        rows.append({"key": f"v:{v['name']}|{v['district']}",
                     "name": v["name"], "district": v["district"],
                     "distance_km": round(float(d[j]), 2),
                     "nearest_named": gname[idx[j]] or None})

    resolved = [r for r in rows if r["distance_km"] is not None]
    save_json(OUT, {
        "generated_at": now,
        "class": OBSERVED,
        "source": SRC,
        "note": ("Distance to the nearest mapped river CHANNEL, which is what a "
                 "reader means by 'from the river'. Distinct from the distance to "
                 "this project's modelled GloFAS point for the basin, which can be "
                 "tens of km away along the same river."),
        "villages_total": len(rows),
        "villages_resolved": len(resolved),
        "tiles_missing": [f"{a:+.1f},{b:+.1f}" for a, b in missing],
        "villages": rows,
    })
    print(f"OK data/river_distance.json: {len(resolved)}/{len(rows)} villages resolved"
          + (f", {len(missing)} tile(s) missing - rerun to fill" if missing else ""))
    return 0 if len(resolved) == len(rows) and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
