"""Sentinel-1 observed impact on the mapped villages.

The point of Sentinel-1 for this project is not a pretty overlay: it is the
one source that MEASURES where water actually was, over the very villages
the map is about. This step intersects each village with the S1-derived
water footprint of each real flood event and states, as OBSERVED fact, which
settlements fell inside the observed water.

Honesty rules (non-negotiable):
- A village is reported "in observed water" ONLY if it lies inside the
  water polygons AND inside the scene's processed area. Villages outside
  the processed box are "not observed by this scene" - NEVER "dry". The
  three states (in_water / observed_dry / outside_observed_area) are kept
  distinct so absence of evidence is never dressed up as evidence.
- The S1 extent is SURFACE WATER at acquisition time and includes permanent
  rivers/wetlands (no pre-event differencing). So "in observed water" means
  the village sat within observed surface water on that date - it does NOT
  by itself prove new flooding. That caveat travels with every count.
- This is a post-event OBSERVATION, not a model score. We did not predict
  these events; grading predictions against S1 begins with the forward
  ledger (archive_model_fc.py), not retroactively.
- Class OBSERVED. Nothing here is fabricated: if an extent is missing or
  degraded, its event is reported as unavailable with the reason.

Reads:  data/villages.json, public/flood_extent_*.geojson,
        public/flood_extents_status.json
Writes: public/s1_village_impact.json
"""
from __future__ import annotations
import sys

from common import OBSERVED, DATA_DIR, PUBLIC_DIR, load_json, save_json, utc_now_iso

# event -> extent file (same mapping as s1_water_extent.py)
EVENTS = [
    ("2022-06 Silchar", "flood_extent_2022.geojson"),
    ("2026-07 Upper Assam", "flood_extent_2026.geojson"),
]


def point_in_ring(lon: float, lat: float, ring: list) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (y1 > lat) != (y2 > lat):
            xin = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < xin:
                inside = not inside
    return inside


def in_multipolygon(lon: float, lat: float, coords: list) -> bool:
    """coords is a MultiPolygon coordinate array: [ [outer_ring, *holes], ... ].
    A point counts as inside if it is in any polygon's outer ring and not in
    one of that polygon's holes."""
    for poly in coords:
        if not poly:
            continue
        if point_in_ring(lon, lat, poly[0]) and not any(
                point_in_ring(lon, lat, hole) for hole in poly[1:]):
            return True
    return False


def in_bbox(lon: float, lat: float, bbox: list) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def assess_event(villages: list, extent_path) -> dict:
    doc = load_json(extent_path)
    feat = doc["features"][0]
    props = feat["properties"]
    coords = feat["geometry"]["coordinates"]
    box = props.get("processed_bbox") or props.get("aoi_bbox")

    in_water, dry = [], 0
    observed = 0
    for v in villages:
        lon, lat = v["lon"], v["lat"]
        if box and not in_bbox(lon, lat, box):
            continue                                   # not seen by this scene
        observed += 1
        if in_multipolygon(lon, lat, coords):
            in_water.append({"name": v["name"], "district": v.get("district"),
                             "lat": round(lat, 5), "lon": round(lon, 5)})
        else:
            dry += 1
    in_water.sort(key=lambda x: (x.get("district") or "", x["name"]))
    return {
        "status": "assessed",
        "scene_id": props.get("scene_id"),
        "acquired": props.get("acquired"),
        "processed_bbox": box,
        "villages_in_processed_area": observed,
        "villages_in_observed_water": len(in_water),
        "villages_observed_dry": dry,
        "in_water": in_water,
        "class": OBSERVED,
        "caveat": ("S1 extent is surface water at acquisition time and includes "
                   "permanent rivers/wetlands (no pre-event differencing): a "
                   "village 'in observed water' sat within observed surface water "
                   "on that date and is not by itself proof of new flooding. "
                   "Villages outside the processed box were not observed by this "
                   "scene, not necessarily dry."),
        "source": "intersection of OSM village points with Sentinel-1 water extent (this repo)",
    }


def main() -> int:
    now = utc_now_iso()
    villages = load_json(DATA_DIR / "villages.json")["villages"]
    events = {}
    for name, fname in EVENTS:
        path = PUBLIC_DIR / fname
        if not path.exists():
            events[name] = {"status": "unavailable", "reason": f"{fname} not present"}
            print(f"  {name}: extent file missing - reported unavailable")
            continue
        try:
            events[name] = assess_event(villages, path)
            e = events[name]
            print(f"  {name}: {e['villages_in_observed_water']} of "
                  f"{e['villages_in_processed_area']} mapped villages in the "
                  f"processed area fell within observed water "
                  f"(scene {str(e['scene_id'])[:32]}..., {str(e['acquired'])[:10]})")
        except (KeyError, IndexError, ValueError) as err:
            events[name] = {"status": "unavailable", "reason": str(err)}
            print(f"  {name}: could not assess - {err}")

    save_json(PUBLIC_DIR / "s1_village_impact.json", {
        "generated_at": now,
        "note": ("Which mapped villages fell inside the Sentinel-1 observed water "
                 "footprint of each flood event. OBSERVED fact, not a model score; "
                 "surface water includes permanent rivers (see per-event caveat)."),
        "village_count_total": len(villages),
        "events": events,
    })
    print(f"OK s1 village impact -> public/s1_village_impact.json "
          f"({len(villages)} villages checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
