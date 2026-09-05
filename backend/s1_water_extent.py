"""UPGRADE v2, STEP B2 - Sentinel-1 water extent for both flood events.

For each event, picks ONE peak scene from public/s1_footprints.geojson,
processes a SMALL area of interest through the Copernicus Data Space
openEO API (sar_backscatter -> sigma0, resampled to ~55 m, GeoTIFF), then
thresholds VV sigma0 locally and polygonises the water mask.

  Event "2022-06 Silchar"     -> public/flood_extent_2022.geojson
  Event "2026-07 Upper Assam" -> public/flood_extent_2026.geojson

Auth: CDSE_USERNAME / CDSE_PASSWORD environment variables (free account,
https://dataspace.copernicus.eu), password grant against the CDSE identity
service - the same variables documented in s1_water_extent_stretch.py.

Honesty rules (non-negotiable):
- The derived mask is real measurement, class OBSERVED, with scene ID,
  acquisition date, method and threshold in the feature properties.
- The -18 dB VV threshold itself is an UNVALIDATED demo choice and says so.
- The mask is SURFACE WATER at acquisition time - it includes permanent
  rivers/wetlands (no pre-event differencing) and says so.
- If auth or processing genuinely fails after retries: footprints only,
  the exact error goes into public/flood_extents_status.json (and
  the project's internal records), and NOTHING is fabricated.

Bulky intermediates (GeoTIFFs) are deleted after processing.
Idempotent: skips an event whose extent file already exists (--refresh to
redo). Always exits 0 unless the footprints file itself is missing.
"""

import datetime
import os
import shutil
import sys
import time

import requests

from common import (OBSERVED, PUBLIC_DIR, RAW_DIR, load_json, save_json,
                    utc_now_iso)

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")
OPENEO = "https://openeo.dataspace.copernicus.eu/openeo/1.2"
BATCH_POLL_S = 15
BATCH_MAX_S = 900
# CDSE's sar_backscatter (Orfeo) currently fails on Sentinel-1D products
# (verified 2026-08-28: S1D job -> "Too many soft errors", S1A job -> OK).
# Prefer other platforms; if only S1D covers the AOI we still try and
# report the server's real error.
PLATFORM_AVOID = ("sentinel-1d",)

THRESHOLD_DB = -18.0
THRESHOLD_LIN = 10 ** (THRESHOLD_DB / 10.0)          # sigma0 < 0.0158 -> water
BLOCK = 6                                             # 6 x 10 m native -> 60 m cells
MAX_RECTANGLES = 9000

TMP_DIR = RAW_DIR / "s1_tmp"

# AOIs sit where the tracked villages are. The only 2022 peak-flood pass
# (23 Jun) covers Cachar east of ~92.86 only, so the 2022 AOI reaches east
# over the Sonai/Udharbond/Lakhipur floodplain; the processed area is always
# clipped to what the chosen scene actually observed (see pick_scene).
EVENTS = [
    {"event": "2022-06 Silchar", "out": "flood_extent_2022.geojson",
     "aoi": [92.65, 24.70, 93.30, 24.95],       # Barak floodplain, Silchar to Lakhipur/Jirighat
     "prefer_after": "2022-06-20"},
    {"event": "2026-07 Upper Assam", "out": "flood_extent_2026.geojson",
     "aoi": [94.45, 26.85, 94.95, 27.15],       # Sivasagar/Charaideo, Dikhow/Disang
     "prefer_after": "2026-07-19"},
]

METHOD = (f"VV sigma0-ellipsoid backscatter threshold < {THRESHOLD_DB:.0f} dB "
          f"(openEO sar_backscatter on CDSE at native 10 m UTM, block-averaged "
          f"locally to {BLOCK * 10} m, thresholded; threshold is an UNVALIDATED "
          f"demo choice)")


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


def footprint_contains(geom: dict, lon: float, lat: float) -> bool:
    if geom["type"] == "Polygon":
        return point_in_ring(lon, lat, geom["coordinates"][0])
    if geom["type"] == "MultiPolygon":
        return any(point_in_ring(lon, lat, poly[0]) for poly in geom["coordinates"])
    return False


def geom_bbox(geom: dict) -> tuple:
    rings = ([geom["coordinates"][0]] if geom["type"] == "Polygon"
             else [poly[0] for poly in geom["coordinates"]])
    xs = [c[0] for r in rings for c in r]
    ys = [c[1] for r in rings for c in r]
    return min(xs), min(ys), max(xs), max(ys)


def pick_scene(features: list, ev: dict) -> tuple:
    """Deterministic pick over this event's peak-window scenes.
    Returns (scene, effective_aoi, note) or (None, None, reason).
    Preference: scene whose footprint contains ALL four AOI corners
    (earliest on/after prefer_after, else earliest). Fallback: scene with
    the largest bbox overlap; the processed AOI is then CLIPPED to that
    overlap (inset 0.02 deg from the scene edge) - we only ever threshold
    what the scene actually observed."""
    aoi = ev["aoi"]
    peaks = [f for f in features
             if f["properties"]["event"] == ev["event"]
             and f["properties"]["window"] == "peak_flood"]
    if not peaks:
        return None, None, "no peak-window scenes for this event"
    corners = [(aoi[0], aoi[1]), (aoi[2], aoi[1]), (aoi[2], aoi[3]), (aoi[0], aoi[3])]
    full = [f for f in peaks
            if all(footprint_contains(f["geometry"], x, y) for x, y in corners)]
    def platform_ok(f):
        return str(f["properties"].get("platform", "")).lower() not in PLATFORM_AVOID

    if full:
        full.sort(key=lambda f: (not platform_ok(f), f["properties"]["acquired"]))
        after = [f for f in full if f["properties"]["acquired"][:10] >= ev["prefer_after"]]
        pick = (after or full)[0]
        note = "scene covers the full AOI" + ("" if platform_ok(pick) else
                                              " (only Sentinel-1D scenes available - CDSE backscatter may fail)")
        return pick, aoi, note

    def overlap(f):
        bx = geom_bbox(f["geometry"])
        w = min(aoi[2], bx[2]) - max(aoi[0], bx[0])
        h = min(aoi[3], bx[3]) - max(aoi[1], bx[1])
        return max(w, 0.0) * max(h, 0.0)

    peaks.sort(key=lambda f: (-overlap(f), f["properties"]["acquired"]))
    best = peaks[0]
    if overlap(best) < 0.02 * 0.02:
        return None, None, "no peak-window scene overlaps the AOI"
    # inscribed rectangle: for each sampled latitude find the inside-lon span,
    # for each sampled longitude the inside-lat span; the intersection of
    # those spans is an axis-aligned box that lies INSIDE the footprint quad
    # (bbox-based clipping can fall outside a rotated GRD footprint)
    def frange(a, b, n):
        return [a + (b - a) * i / (n - 1) for i in range(n)]
    lon_s = frange(aoi[0], aoi[2], 120)
    lat_s = frange(aoi[1], aoi[3], 60)
    west, east = aoi[0], aoi[2]
    for lat in lat_s:
        ins = [x for x in lon_s if footprint_contains(best["geometry"], x, lat)]
        if not ins:
            continue
        west, east = max(west, min(ins)), min(east, max(ins))
    south, north = aoi[1], aoi[3]
    for lon in frange(west, east, 60) if east > west else []:
        ins = [y for y in lat_s if footprint_contains(best["geometry"], lon, y)]
        if not ins:
            continue
        south, north = max(south, min(ins)), min(north, max(ins))
    eff = [west + 0.01, south + 0.01, east - 0.01, north - 0.01]      # safety inset
    corners_ok = all(footprint_contains(best["geometry"], x, y)
                     for x, y in ((eff[0], eff[1]), (eff[2], eff[1]), (eff[2], eff[3]), (eff[0], eff[3])))
    if eff[2] - eff[0] < 0.05 or eff[3] - eff[1] < 0.05 or not corners_ok:
        return None, None, "no peak-window scene contains a usable part of the AOI"
    note = (f"scene covers the AOI only partially; processed area clipped to the part "
            f"inside the scene footprint ({eff[0]:.2f},{eff[1]:.2f})..({eff[2]:.2f},{eff[3]:.2f})")
    return best, [round(v, 3) for v in eff], note


def get_token() -> str:
    user = os.environ.get("CDSE_USERNAME")
    pwd = os.environ.get("CDSE_PASSWORD")
    if not (user and pwd):
        raise RuntimeError(
            "CDSE_USERNAME / CDSE_PASSWORD not set in the environment "
            "(checked process env; free account at https://dataspace.copernicus.eu)")
    last = None
    for attempt in range(3):
        try:
            resp = requests.post(TOKEN_URL, data={
                "grant_type": "password", "client_id": "cdse-public",
                "username": user, "password": pwd}, timeout=60)
            resp.raise_for_status()
            return resp.json()["access_token"]
        except Exception as err:  # noqa: BLE001 - retry loop
            last = err
            time.sleep(2 * (2 ** attempt))
    raise RuntimeError(f"CDSE token request failed after 3 attempts: {last}")


def openeo_sigma0_tiff(token: str, aoi: list, day: str, out_path) -> None:
    """Sync openEO job: SENTINEL1_GRD VV -> sigma0 -> ~55 m GTiff for one day."""
    nxt = (datetime.date.fromisoformat(day) + datetime.timedelta(days=1)).isoformat()
    graph = {
        "load": {"process_id": "load_collection", "arguments": {
            "id": "SENTINEL1_GRD",
            "spatial_extent": {"west": aoi[0], "south": aoi[1],
                               "east": aoi[2], "north": aoi[3]},
            "temporal_extent": [day, nxt],
            "bands": ["VV"]}},
        "sar": {"process_id": "sar_backscatter", "arguments": {
            "data": {"from_node": "load"}, "coefficient": "sigma0-ellipsoid"}},
        # NOTE: no resample_spatial here - verified 2026-08-28 that a
        # reprojection in the graph is pushed ahead of the backscatter step
        # and yields an EMPTY raster ("Layout extent split in 0 tiles").
        # We take the native UTM 10 m product and downsample locally.
        "reduce": {"process_id": "reduce_dimension", "arguments": {
            "data": {"from_node": "sar"}, "dimension": "t",
            "reducer": {"process_graph": {"m": {
                "process_id": "mean",
                "arguments": {"data": {"from_parameter": "data"}},
                "result": True}}}}},
        "save": {"process_id": "save_result", "arguments": {
            "data": {"from_node": "reduce"}, "format": "GTiff"}, "result": True},
    }
    # BATCH job, not the synchronous endpoint: sync silently returned an
    # empty raster when the server-side backscatter step failed (verified
    # 2026-08-28); a batch job exposes status + logs so failures are real.
    H = {"Authorization": f"Bearer oidc/CDSE/{token}"}
    r = requests.post(f"{OPENEO}/jobs", headers=H, timeout=60,
                      json={"process": {"process_graph": graph},
                            "title": f"jaldrishti-water-extent-{day}"})
    r.raise_for_status()
    job = r.headers.get("OpenEO-Identifier") or r.json().get("id")
    if not job:
        raise RuntimeError("openEO job creation returned no job id")
    requests.post(f"{OPENEO}/jobs/{job}/results", headers=H, timeout=60).raise_for_status()
    status, waited = "queued", 0
    while waited < BATCH_MAX_S:
        time.sleep(BATCH_POLL_S)
        waited += BATCH_POLL_S
        status = requests.get(f"{OPENEO}/jobs/{job}", headers=H, timeout=60).json().get("status")
        if status in ("finished", "error", "canceled"):
            break
    if status != "finished":
        logs = requests.get(f"{OPENEO}/jobs/{job}/logs", headers=H, timeout=60).json().get("logs", [])
        errs = [l.get("message", "") for l in logs if l.get("level") == "error"]
        detail = (errs[-1] if errs else f"status {status} after {waited}s")[:300]
        raise RuntimeError(f"openEO job {job} {status}: {detail}")
    res = requests.get(f"{OPENEO}/jobs/{job}/results", headers=H, timeout=60).json()
    assets = [a for a in res.get("assets", {}).values()
              if "tif" in str(a.get("type", "")).lower() or str(a.get("href", "")).lower().endswith(".tif")]
    if not assets:
        raise RuntimeError(f"openEO job {job} finished but produced no GeoTIFF asset")
    href = assets[0]["href"]
    content = None
    for hdrs in (H, {}):                       # signed hrefs may reject the bearer header
        rr = requests.get(href, headers=hdrs, timeout=600)
        if rr.status_code == 200 and rr.content[:4] in (b"II*\x00", b"MM\x00*"):
            content = rr.content
            break
    if content is None:
        raise RuntimeError(f"openEO result download is not a TIFF (HTTP {rr.status_code}: "
                           f"{rr.content[:120]!r})")
    out_path.write_bytes(content)


def read_geotiff(path) -> tuple:
    """(array, x0, y0, sx, sy, epsg) via tifffile - Pillow mis-decodes these
    tiled float32 GeoTIFFs (verified 2026-08-28); no rasterio needed."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = {t.name: t.value for t in page.tags.values()}
        scale = tags.get("ModelPixelScaleTag")
        tie = tags.get("ModelTiepointTag")
        if scale is None or tie is None:
            raise RuntimeError("GeoTIFF lacks ModelPixelScale/ModelTiepoint tags")
        epsg = None
        keys = tags.get("GeoKeyDirectoryTag")
        if keys is not None:
            k = list(keys)
            for i in range(4, len(k) - 3, 4):
                if k[i] == 3072:          # ProjectedCSTypeGeoKey
                    epsg = int(k[i + 3])
        arr = page.asarray().astype("float32")
    sx, sy = float(scale[0]), float(scale[1])
    x0 = float(tie[3]) - float(tie[0]) * sx
    y0 = float(tie[4]) + float(tie[1]) * sy
    return arr, x0, y0, sx, sy, epsg


def utm_to_lonlat(x, y, epsg: int):
    """Inverse transverse Mercator, WGS84 (numpy arrays), for UTM north zones
    EPSG:326xx. Accuracy ~1 cm; avoids a pyproj dependency."""
    import numpy as np
    zone = epsg - 32600
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    a, f = 6378137.0, 1 / 298.257223563
    k0, e2 = 0.9996, 2 * f - f * f
    ep2 = e2 / (1 - e2)
    x = (np.asarray(x, dtype="float64") - 500000.0) / k0
    M = np.asarray(y, dtype="float64") / k0
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    sp, cp, tp = np.sin(phi1), np.cos(phi1), np.tan(phi1)
    N1 = a / np.sqrt(1 - e2 * sp ** 2)
    T1, C1 = tp ** 2, ep2 * cp ** 2
    R1 = a * (1 - e2) / (1 - e2 * sp ** 2) ** 1.5
    D = x / N1
    lat = phi1 - (N1 * tp / R1) * (D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * D ** 4 / 24
                                   + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = lon0 + (D - (1 + 2 * T1 + C1) * D ** 3 / 6
                  + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2) * D ** 5 / 120) / cp
    return np.degrees(lon), np.degrees(lat)


def mask_to_rectangles(mask, x0, y0, sx, sy, epsg=None) -> list:
    """Row-run rectangles from a boolean mask; corners converted from the
    native grid (UTM if epsg given) to lon/lat. Coarsens the mask 2x at a
    time if the rectangle count would exceed MAX_RECTANGLES."""
    import numpy as np
    factor = 1
    while True:
        runs = []                       # (x1, y1, x2, y2) in native units
        h, w = mask.shape
        for i in range(h):
            row = mask[i]
            j = 0
            while j < w:
                if row[j]:
                    j2 = j
                    while j2 < w and row[j2]:
                        j2 += 1
                    runs.append((x0 + j * sx * factor, y0 - (i + 1) * sy * factor,
                                 x0 + j2 * sx * factor, y0 - i * sy * factor))
                    j = j2
                else:
                    j += 1
        if len(runs) <= MAX_RECTANGLES or factor >= 8:
            break
        h2, w2 = (mask.shape[0] // 2) * 2, (mask.shape[1] // 2) * 2
        m = mask[:h2, :w2]
        mask = m[0::2, 0::2] | m[1::2, 0::2] | m[0::2, 1::2] | m[1::2, 1::2]
        factor *= 2
    if not runs:
        return []
    r = np.array(runs, dtype="float64")
    if epsg and 32601 <= epsg <= 32660:
        xs = np.concatenate([r[:, 0], r[:, 2], r[:, 2], r[:, 0]])
        ys = np.concatenate([r[:, 1], r[:, 1], r[:, 3], r[:, 3]])
        lon, lat = utm_to_lonlat(xs, ys, epsg)
        n = len(r)
        c = [(lon[k], lat[k]) for k in range(4 * n)]
        return [[[[round(c[k][0], 5), round(c[k][1], 5)],
                  [round(c[n + k][0], 5), round(c[n + k][1], 5)],
                  [round(c[2 * n + k][0], 5), round(c[2 * n + k][1], 5)],
                  [round(c[3 * n + k][0], 5), round(c[3 * n + k][1], 5)],
                  [round(c[k][0], 5), round(c[k][1], 5)]]] for k in range(n)]
    return [[[[round(a, 5), round(b, 5)], [round(cc, 5), round(b, 5)],
              [round(cc, 5), round(d, 5)], [round(a, 5), round(d, 5)],
              [round(a, 5), round(b, 5)]]] for a, b, cc, d in runs]


def derive_event(token: str, ev: dict, scene: dict, aoi_eff: list,
                 cover_note: str, retrieved_at: str) -> dict:
    import numpy as np
    props = scene["properties"]
    day = props["acquired"][:10]
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tiff = TMP_DIR / f"sigma0_{ev['out']}.tif"
    try:
        openeo_sigma0_tiff(token, aoi_eff, day, tiff)
        arr, x0, y0, sx, sy, epsg = read_geotiff(tiff)
        # block-average to BLOCK x native resolution (speckle reduction + size)
        h, w = (arr.shape[0] // BLOCK) * BLOCK, (arr.shape[1] // BLOCK) * BLOCK
        blk = arr[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
        with np.errstate(invalid="ignore"):
            arr = np.nanmean(blk, axis=(1, 3))
        sx, sy = sx * BLOCK, sy * BLOCK
        valid = np.isfinite(arr) & (arr > 1e-12)      # denormal/zero fill = no data
        n_valid = int(valid.sum())
        # sanity gate: an empty or constant raster must NEVER become a polygon
        if n_valid < 0.05 * arr.size:
            raise RuntimeError(f"raster has {n_valid}/{arr.size} valid pixels - "
                               f"openEO returned no imagery for this AOI/date")
        vals = arr[valid]
        if float(np.nanstd(vals)) < 1e-9 or float(np.nanmedian(vals)) > 10:
            raise RuntimeError(f"raster is degenerate (std={float(np.nanstd(vals)):.3g}, "
                               f"median={float(np.nanmedian(vals)):.3g}) - units/empty check failed")
        mask = valid & (arr < THRESHOLD_LIN)
        water_frac = float(mask.sum()) / n_valid
        if water_frac > 0.90:
            raise RuntimeError(f"{100 * water_frac:.0f}% of valid pixels classified as water - "
                               f"implausible, refusing to ship")
        rects = mask_to_rectangles(mask, x0, y0, sx, sy, epsg)
        water_stats = {"epsg": epsg, "cell_m": sx, "valid_pixels": n_valid,
                       "median_sigma0": round(float(np.nanmedian(vals)), 4)}
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)   # bulky intermediates gone

    feature = {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": rects},
        "properties": {
            "event": ev["event"],
            "scene_id": props["scene_id"],
            "acquired": props["acquired"],
            "platform": props.get("platform"),
            "orbit_state": props.get("orbit_state"),
            "method": METHOD,
            "threshold_db": THRESHOLD_DB,
            "aoi_bbox": ev["aoi"],
            "processed_bbox": aoi_eff,
            "coverage": cover_note,
            "water_fraction_of_aoi": round(water_frac, 4),
            "raster": water_stats,
            "note": ("Surface water at acquisition time - includes permanent "
                     "rivers and wetlands (no pre-event differencing). The "
                     "threshold is an unvalidated demo choice."),
            "class": OBSERVED,
            "source": ("Sentinel-1 GRD via CDSE openEO (sar_backscatter "
                       "sigma0-ellipsoid), thresholded locally"),
            "retrieved_at": retrieved_at,
        },
    }
    doc = {"type": "FeatureCollection",
           "generated_at": retrieved_at,
           "note": feature["properties"]["note"],
           "features": [feature]}
    save_json(PUBLIC_DIR / ev["out"], doc)
    print(f"  {ev['event']}: {len(rects)} water rectangles, "
          f"{100 * water_frac:.1f}% of AOI, scene {props['scene_id'][:44]}...")
    return {"status": "shipped", "file": ev["out"], "scene_id": props["scene_id"],
            "acquired": props["acquired"], "attempted_at": retrieved_at}


def main() -> int:
    retrieved_at = utc_now_iso()
    fp = load_json(PUBLIC_DIR / "s1_footprints.geojson")   # hard requirement
    status = {}
    token = None
    token_err = None

    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    prev_status = (load_json(PUBLIC_DIR / "flood_extents_status.json").get("events", {})
                   if (PUBLIC_DIR / "flood_extents_status.json").exists() else {})
    for ev in EVENTS:
        if only and only not in ev["event"]:
            if ev["event"] in prev_status:
                status[ev["event"]] = prev_status[ev["event"]]   # keep last known state
            continue
        out_path = PUBLIC_DIR / ev["out"]
        if out_path.exists() and "--refresh" not in sys.argv:
            prev = load_json(out_path)
            p = prev["features"][0]["properties"]
            status[ev["event"]] = {"status": "shipped", "file": ev["out"],
                                   "scene_id": p["scene_id"], "acquired": p["acquired"],
                                   "attempted_at": p["retrieved_at"],
                                   "note": "already present (static history); --refresh to redo"}
            print(f"  {ev['event']}: extent already present, skipping")
            continue

        scene, aoi_eff, cover_note = pick_scene(fp["features"], ev)
        if scene is None:
            status[ev["event"]] = {"status": "degraded", "attempted_at": retrieved_at,
                                   "error": cover_note}
            print(f"  {ev['event']}: DEGRADED - {cover_note}")
            continue
        print(f"  {ev['event']}: scene {scene['properties']['scene_id'][:44]}... "
              f"({scene['properties']['acquired'][:16]}) - {cover_note}")

        if token is None and token_err is None:
            try:
                token = get_token()
            except RuntimeError as err:
                token_err = str(err)
        if token is None:
            status[ev["event"]] = {"status": "degraded", "attempted_at": retrieved_at,
                                   "candidate_scene": scene["properties"]["scene_id"],
                                   "error": token_err}
            print(f"  {ev['event']}: DEGRADED - {token_err}")
            continue

        try:
            status[ev["event"]] = derive_event(token, ev, scene, aoi_eff,
                                               cover_note, retrieved_at)
        except (RuntimeError, OSError, ValueError) as err:
            status[ev["event"]] = {"status": "degraded", "attempted_at": retrieved_at,
                                   "candidate_scene": scene["properties"]["scene_id"],
                                   "error": str(err)}
            print(f"  {ev['event']}: DEGRADED - {err}")

    save_json(PUBLIC_DIR / "flood_extents_status.json", {
        "generated_at": retrieved_at,
        "note": ("Status of the Step B2 water-extent derivation per event. "
                 "'degraded' means the derivation could not run (exact error "
                 "recorded) and NO extent was fabricated - footprints still ship."),
        "events": status,
    })

    n_shipped = sum(1 for s in status.values() if s["status"] == "shipped")
    print(f"OK flood extents: {n_shipped}/{len(EVENTS)} shipped, "
          f"status -> public/flood_extents_status.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
