"""STEP 5 (extended by Upgrade v2, STEP B1) - Sentinel-1 GRD scene
footprints for TWO real flood events (REAL, HISTORICAL).

Source: Copernicus Data Space Ecosystem STAC catalogue
        https://stac.dataspace.copernicus.eu/v1  (search is public, no auth)

Event 1 - "2022-06 Silchar": June 2022 Barak flood, Cachar district.
  pre_flood  2022-05-10 .. 2022-05-25, peak_flood 2022-06-20 .. 2022-06-25.
Event 2 - "2026-07 Upper Assam": the floods that followed the 19 July 2026
  Mon-district cloudburst, over Sivasagar/Charaideo/Jorhat.
  pre_flood  2026-06-15 .. 2026-06-30 (pre-event baseline),
  peak_flood 2026-07-15 .. 2026-08-05 (event window).

Output: public/s1_footprints.geojson - scene footprints with acquisition
timestamps, orbit/mode metadata, thumbnail + product-download URLs, and an
"event" tag so the viewer can separate the two events. Class OBSERVED
(real acquisitions). Water-extent derivation is Step B2
(backend/s1_water_extent.py); nothing is fabricated here.
"""

import sys

import requests

from common import OBSERVED, PUBLIC_DIR, RAW_DIR, save_json, utc_now_iso

STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"

EVENTS = [
    {
        "key": "2022_silchar",
        "event": "2022-06 Silchar",
        "event_label": "June 2022 Barak flood (Silchar / Cachar)",
        "bbox": [92.5, 24.6, 93.0, 25.0],
        "windows": [
            {"key": "pre_flood", "label": "pre-flood (mid-May 2022)",
             "datetime": "2022-05-10T00:00:00Z/2022-05-25T23:59:59Z"},
            {"key": "peak_flood", "label": "peak flood (20-25 June 2022)",
             "datetime": "2022-06-20T00:00:00Z/2022-06-25T23:59:59Z"},
        ],
    },
    {
        "key": "2026_upperassam",
        "event": "2026-07 Upper Assam",
        "event_label": "July 2026 Upper Assam floods (Sivasagar / Charaideo / Jorhat)",
        "bbox": [94.0, 26.5, 95.2, 27.2],
        "windows": [
            {"key": "pre_flood", "label": "pre-event (late June 2026)",
             "datetime": "2026-06-15T00:00:00Z/2026-06-30T23:59:59Z"},
            {"key": "peak_flood", "label": "event window (15 July - 5 August 2026)",
             "datetime": "2026-07-15T00:00:00Z/2026-08-05T23:59:59Z"},
        ],
    },
]


def search_window(bbox: list, win: dict) -> list[dict]:
    """Paginated STAC GET search for sentinel-1-grd items in one window."""
    items, url = [], STAC_SEARCH
    params = {"collections": "sentinel-1-grd", "bbox": ",".join(map(str, bbox)),
              "datetime": win["datetime"], "limit": 50}
    for _ in range(10):  # pagination safety cap
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        page = resp.json()
        items.extend(page.get("features", []))
        nxt = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            break
        url, params = nxt["href"], {}  # next link is self-contained
    return items


def download_thumb(url: str, scene_id: str) -> str | None:
    """Cache the public quicklook PNG locally (CDSE serves it without auth,
    but browser-side CORS makes a local copy the reliable demo path)."""
    out_dir = PUBLIC_DIR / "s1_thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scene_id}.png"
    if not out.exists():
        try:
            resp = requests.get(url, timeout=60, allow_redirects=True)
            resp.raise_for_status()
            out.write_bytes(resp.content)
        except requests.RequestException:
            return None
    return f"s1_thumbs/{scene_id}.png"


def main() -> int:
    retrieved_at = utc_now_iso()
    features = []
    n_thumbs = 0

    for ev in EVENTS:
        for win in ev["windows"]:
            items = search_window(ev["bbox"], win)
            save_json(RAW_DIR / f"stac_s1_{ev['key']}_{win['key']}.json",
                      {"event": ev["event"], "window": win, "count": len(items),
                       "features": items})
            for it in items:
                props = it.get("properties", {})
                assets = it.get("assets", {})
                thumb = assets.get("thumbnail", {}).get("href")
                product = assets.get("Product", {}).get("href")
                local_thumb = download_thumb(thumb, it["id"]) if thumb else None
                n_thumbs += bool(local_thumb)
                features.append({
                    "type": "Feature",
                    "geometry": it["geometry"],
                    "properties": {
                        "scene_id": it["id"],
                        "event": ev["event"],
                        "event_label": ev["event_label"],
                        "window": win["key"],
                        "window_label": win["label"],
                        "acquired": props.get("datetime"),
                        "platform": props.get("platform"),
                        "instrument_mode": props.get("sar:instrument_mode"),
                        "orbit_state": props.get("sat:orbit_state"),
                        "polarizations": props.get("sar:polarizations"),
                        "thumbnail_href": thumb,
                        "local_thumbnail": local_thumb,
                        "product_download_href": product,
                        "class": OBSERVED,
                        "source": "Copernicus Data Space Ecosystem STAC catalogue (sentinel-1-grd)",
                        "retrieved_at": retrieved_at,
                    },
                })
            print(f"  {ev['key']:16s} {win['key']:10s}: {len(items)} scenes")

    doc = {
        "type": "FeatureCollection",
        "note": ("Real Sentinel-1 GRD acquisitions for two flood events: the June 2022 "
                 "Barak flood over Cachar/Silchar, and the July 2026 Upper Assam floods "
                 "that followed the 19 July 2026 cloudburst in Mon district, Nagaland "
                 "(a real disaster with loss of life; treated factually). Footprints + "
                 "metadata from the public CDSE STAC catalogue; each feature carries an "
                 "'event' tag. Quicklook thumbnails are public and cached locally under "
                 "public/s1_thumbs/. Water-extent derivation from pixel data is "
                 "backend/s1_water_extent.py (Step B2); nothing is fabricated here."),
        "generated_at": retrieved_at,
        "events": [{"event": ev["event"], "event_label": ev["event_label"],
                    "bbox_searched": ev["bbox"]} for ev in EVENTS],
        "features": features,
    }
    save_json(PUBLIC_DIR / "s1_footprints.geojson", doc)

    # ---- verification ----
    check = __import__("json").load(open(PUBLIC_DIR / "s1_footprints.geojson", encoding="utf-8"))
    seen = {(f["properties"]["event"], f["properties"]["window"]) for f in check["features"]}
    for ev in EVENTS:
        for win in ev["windows"]:
            assert (ev["event"], win["key"]) in seen, f"missing {ev['event']} / {win['key']}"
    for f in check["features"]:
        assert f["geometry"]["type"] in ("Polygon", "MultiPolygon")
        assert f["properties"]["class"] == OBSERVED
        assert f["properties"]["acquired"] and f["properties"]["scene_id"]
        assert f["properties"]["event"] in {e["event"] for e in EVENTS}
    n_2026 = sum(1 for f in check["features"] if f["properties"]["event"] == "2026-07 Upper Assam")
    print(f"OK public/s1_footprints.geojson: {len(check['features'])} scenes "
          f"({n_2026} for 2026 event), all event/window combos present, "
          f"{n_thumbs} quicklooks cached locally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
