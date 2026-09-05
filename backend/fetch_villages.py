"""STEP 3 - build data/villages.json: ~50 REAL village/settlement names across
Sivasagar, Morigaon and Cachar districts (Assam).

Names are real settlements the author is confident exist in these districts.
Coordinates: OSM Overpass is attempted first (coordinate_precision "osm_node",
source "OpenStreetMap via Overpass API"); on miss or Overpass failure we fall
back to approximate coordinates from general knowledge
(coordinate_precision "approximate", source "estimated").

Points only, no polygons. No LGD-code claims. Demo subset, not a roster.
"""

import json
import time
import sys
import unicodedata

import requests

from common import DATA_DIR, RAW_DIR, save_json, utc_now_iso


def norm(s: str) -> str:
    """Casefold and strip diacritics so 'Hālwāting' matches 'Halwating'."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().strip()


# known OSM spelling variants for our names
ALIASES = {"demow": ["dimow"], "morigaon": ["marigaon"]}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# District bounding boxes for the Overpass place-node sweep (south,west,north,east)
DISTRICT_BBOX = {
    "Sivasagar": (26.70, 94.30, 27.30, 95.10),
    "Morigaon":  (25.95, 91.95, 26.60, 92.65),
    "Cachar":    (24.50, 92.30, 25.05, 93.25),
    # Phase 1c (generic coverage): one district per remaining river point
    "Sonitpur":  (26.55, 92.40, 27.00, 93.10),   # Jia Bharali / Brahmaputra @ Tezpur
    "Golaghat":  (26.10, 93.50, 26.80, 94.15),   # Dhansiri
    "Charaideo": (26.80, 94.80, 27.10, 95.25),   # Disang
    "Nagaon":    (26.00, 92.40, 26.55, 93.00),   # Kopili @ Kampur
    "Barpeta":   (26.20, 90.70, 26.60, 91.20),   # Beki
    # Goal C: Subansiri (Lakhimpur), Manas (Chirang/Bongaigaon), Guwahati reach (Kamrup)
    "Lakhimpur":  (26.90, 93.80, 27.45, 94.60),
    "Chirang":    (26.30, 90.30, 26.70, 90.80),
    "Bongaigaon": (26.15, 90.45, 26.65, 90.90),
    "Kamrup":     (25.90, 91.10, 26.30, 91.65),
    "Hailakandi": (24.30, 92.40, 24.85, 92.75),   # Katakhal valley (Goal D)
    # coverage fill 2026-09-04: districts inside already-covered basins
    "Dibrugarh":  (27.20, 94.70, 27.60, 95.40),   # Brahmaputra @ Dibrugarh reach
    "Dhemaji":    (27.30, 94.20, 27.75, 94.85),   # north bank: Subansiri / Dibrugarh reach
    "Majuli":     (26.85, 94.00, 27.05, 94.40),   # island, Brahmaputra reaches
    "Jorhat":     (26.55, 94.05, 26.90, 94.40),   # south bank, Dikhow/Brahmaputra side
    "Biswanath":  (26.60, 93.00, 26.95, 93.70),   # north bank, Jia Bharali/Tezpur reach
    "Dhubri":     (25.95, 89.70, 26.35, 90.10),   # Sankosh (coverage-watch)
}

# (name, district, fallback_lat, fallback_lon) - names are real; fallback
# coordinates are approximate (about +/-5 km) and clearly labelled as such.
VILLAGES = [
    # Sivasagar district (Dikhow/Disang floodplain, upper Assam)
    ("Gaurisagar", "Sivasagar", 26.95, 94.57), ("Demow", "Sivasagar", 27.17, 94.75),
    ("Amguri", "Sivasagar", 26.81, 94.52), ("Simaluguri", "Sivasagar", 26.95, 94.83),
    ("Nazira", "Sivasagar", 26.92, 94.73), ("Lakwa", "Sivasagar", 27.00, 94.95),
    ("Namti", "Sivasagar", 26.92, 94.66), ("Nitaipukhuri", "Sivasagar", 27.05, 94.75),
    ("Dikhowmukh", "Sivasagar", 27.05, 94.50), ("Disangmukh", "Sivasagar", 27.02, 94.58),
    ("Khelua", "Sivasagar", 26.90, 94.80), ("Thowra", "Sivasagar", 27.10, 94.80),
    ("Halwating", "Sivasagar", 26.97, 94.70), ("Mathurapur", "Sivasagar", 26.85, 94.65),
    ("Bokota", "Sivasagar", 26.90, 94.72), ("Meteka", "Sivasagar", 27.00, 94.68),
    ("Rajmai", "Sivasagar", 26.88, 94.58),
    # Morigaon district (south-bank Brahmaputra/Kopili floodplain, highly flood-prone)
    ("Mayong", "Morigaon", 26.24, 92.03), ("Bhuragaon", "Morigaon", 26.38, 92.25),
    ("Laharighat", "Morigaon", 26.37, 92.42), ("Moirabari", "Morigaon", 26.36, 92.52),
    ("Dharamtul", "Morigaon", 26.20, 92.24), ("Nellie", "Morigaon", 26.10, 92.20),
    ("Jagiroad", "Morigaon", 26.13, 92.13), ("Jaluguti", "Morigaon", 26.28, 92.40),
    ("Sitajakhala", "Morigaon", 26.17, 92.18), ("Ahatguri", "Morigaon", 26.32, 92.47),
    ("Gagalmari", "Morigaon", 26.30, 92.30), ("Barapujia", "Morigaon", 26.22, 92.45),
    ("Charaibahi", "Morigaon", 26.18, 92.30), ("Mikirbheta", "Morigaon", 26.28, 92.28),
    ("Baghara", "Morigaon", 26.24, 92.38), ("Manaha", "Morigaon", 26.33, 92.35),
    # Cachar district (Barak valley)
    ("Udharbond", "Cachar", 24.87, 92.90), ("Sonai", "Cachar", 24.76, 92.87),
    ("Dholai", "Cachar", 24.70, 92.83), ("Lakhipur", "Cachar", 24.79, 93.01),
    ("Banskandi", "Cachar", 24.86, 92.94), ("Salchapra", "Cachar", 24.81, 92.71),
    ("Borkhola", "Cachar", 24.85, 92.65), ("Katigorah", "Cachar", 24.87, 92.58),
    ("Kalain", "Cachar", 24.92, 92.52), ("Jirighat", "Cachar", 24.79, 93.13),
    ("Fulertal", "Cachar", 24.75, 93.05), ("Palonghat", "Cachar", 24.65, 92.95),
    ("Berenga", "Cachar", 24.82, 92.83), ("Dudhpatil", "Cachar", 24.78, 92.75),
    ("Srikona", "Cachar", 24.90, 92.75), ("Dargakona", "Cachar", 24.69, 92.75),
    ("Masimpur", "Cachar", 24.78, 92.80),
    # Sonitpur (Jia Bharali floodplain / Brahmaputra north bank)
    ("Balipara", "Sonitpur", 26.83, 92.78), ("Rangapara", "Sonitpur", 26.84, 92.66),
    ("Dhekiajuli", "Sonitpur", 26.70, 92.48), ("Thelamara", "Sonitpur", 26.79, 92.57),
    ("Missamari", "Sonitpur", 26.87, 92.60), ("Tezpur", "Sonitpur", 26.63, 92.79),
    ("Bhomoraguri", "Sonitpur", 26.60, 92.82),
    # Golaghat (Dhansiri - July 2026 flood path)
    ("Golaghat", "Golaghat", 26.51, 93.97), ("Numaligarh", "Golaghat", 26.62, 93.72),
    ("Bokakhat", "Golaghat", 26.64, 93.61), ("Dergaon", "Golaghat", 26.70, 93.97),
    ("Sarupathar", "Golaghat", 26.18, 93.87), ("Barpathar", "Golaghat", 26.28, 93.90),
    # Charaideo (Disang - July 2026 flood path)
    ("Sonari", "Charaideo", 27.02, 95.02), ("Sapekhati", "Charaideo", 26.96, 95.13),
    ("Mahmora", "Charaideo", 26.98, 94.90), ("Namtola", "Charaideo", 26.90, 94.95),
    ("Borhat", "Charaideo", 26.88, 95.05),
    # Nagaon (Kopili / Kolong)
    ("Kampur", "Nagaon", 26.15, 92.65), ("Raha", "Nagaon", 26.23, 92.52),
    ("Chaparmukh", "Nagaon", 26.20, 92.54), ("Samaguri", "Nagaon", 26.35, 92.83),
    ("Nagaon", "Nagaon", 26.35, 92.68),
    # Barpeta (Beki)
    ("Barpeta Road", "Barpeta", 26.50, 90.97), ("Sarbhog", "Barpeta", 26.50, 90.88),
    ("Kalgachia", "Barpeta", 26.30, 90.85), ("Barpeta", "Barpeta", 26.32, 91.00),
    ("Howly", "Barpeta", 26.42, 90.98),
    # Goal C - Lakhimpur district (Subansiri floodplain)
    ("North Lakhimpur", "Lakhimpur", 27.24, 94.10), ("Bihpuria", "Lakhimpur", 27.02, 93.92),
    ("Naoboicha", "Lakhimpur", 27.09, 94.02), ("Dhakuakhana", "Lakhimpur", 27.22, 94.42),
    ("Ghilamara", "Lakhimpur", 27.15, 94.30), ("Narayanpur", "Lakhimpur", 27.00, 93.87),
    ("Boginadi", "Lakhimpur", 27.20, 94.15), ("Panigaon", "Lakhimpur", 27.08, 94.15),
    ("Telahi", "Lakhimpur", 27.15, 94.05), ("Kadam", "Lakhimpur", 27.12, 94.25),
    # Goal C - Manas floodplain (Chirang / Bongaigaon)
    ("Bijni", "Chirang", 26.50, 90.70), ("Kajalgaon", "Chirang", 26.51, 90.53),
    ("Basugaon", "Chirang", 26.47, 90.42), ("Runikhata", "Chirang", 26.60, 90.55),
    ("Bengtol", "Chirang", 26.62, 90.63), ("Panbari", "Chirang", 26.45, 90.70),
    ("Abhayapuri", "Bongaigaon", 26.32, 90.68), ("Jogighopa", "Bongaigaon", 26.22, 90.58),
    ("Manikpur", "Bongaigaon", 26.42, 90.68), ("Dangtol", "Bongaigaon", 26.44, 90.60),
    # Goal C - Kamrup district (Brahmaputra @ Guwahati reach, south/west bank)
    ("Hajo", "Kamrup", 26.24, 91.52), ("Sualkuchi", "Kamrup", 26.17, 91.57),
    # Goal D - Hailakandi district (Katakhal/Dhaleswari valley)
    ("Hailakandi", "Hailakandi", 24.68, 92.56), ("Lala", "Hailakandi", 24.55, 92.61),
    ("Katlicherra", "Hailakandi", 24.44, 92.59), ("Algapur", "Hailakandi", 24.75, 92.58),
    ("Katakhal", "Hailakandi", 24.80, 92.63), ("Monacherra", "Hailakandi", 24.50, 92.65),
    # coverage fill - Dibrugarh district (Brahmaputra @ Dibrugarh)
    ("Chabua", "Dibrugarh", 27.47, 95.17), ("Lahowal", "Dibrugarh", 27.45, 95.02),
    ("Tengakhat", "Dibrugarh", 27.30, 95.15), ("Naharkatia", "Dibrugarh", 27.29, 95.34),
    ("Duliajan", "Dibrugarh", 27.37, 95.32), ("Khowang", "Dibrugarh", 27.21, 94.90),
    ("Barbaruah", "Dibrugarh", 27.40, 94.97), ("Moran", "Dibrugarh", 27.17, 94.90),
    # coverage fill - Dhemaji district (north bank)
    ("Dhemaji", "Dhemaji", 27.48, 94.58), ("Silapathar", "Dhemaji", 27.60, 94.72),
    ("Gogamukh", "Dhemaji", 27.36, 94.30), ("Sissiborgaon", "Dhemaji", 27.55, 94.65),
    ("Machkhowa", "Dhemaji", 27.43, 94.45),
    # coverage fill - Majuli island
    ("Garamur", "Majuli", 26.95, 94.21), ("Kamalabari", "Majuli", 26.93, 94.17),
    ("Jengraimukh", "Majuli", 27.00, 94.25),
    # coverage fill - Jorhat district
    ("Jorhat", "Jorhat", 26.75, 94.22), ("Titabar", "Jorhat", 26.60, 94.20),
    ("Mariani", "Jorhat", 26.66, 94.32), ("Teok", "Jorhat", 26.82, 94.35),
    # coverage fill - Biswanath district (Jia Bharali / Tezpur reach north bank)
    ("Biswanath Chariali", "Biswanath", 26.73, 93.15), ("Gohpur", "Biswanath", 26.88, 93.62),
    ("Behali", "Biswanath", 26.82, 93.30), ("Sootea", "Biswanath", 26.75, 93.05),
    # coverage-watch - Dhubri district (Sankosh)
    ("Golokganj", "Dhubri", 26.10, 89.84), ("Agomani", "Dhubri", 26.13, 89.88),
    ("Gauripur", "Dhubri", 26.08, 89.96), ("Dhubri", "Dhubri", 26.02, 89.98),
    ("Tamarhat", "Dhubri", 26.24, 89.87), ("Halakura", "Dhubri", 26.17, 89.93),
    ("Palashbari", "Kamrup", 26.12, 91.54), ("Chhaygaon", "Kamrup", 26.05, 91.39),
    ("Boko", "Kamrup", 25.97, 91.22),
    # Goal C - fill existing districts (9 entries removed 2026-09-03: they
    # duplicated names already in the Phase-1c lists, doubling map counts)
    ("Sarthebari", "Barpeta", 26.37, 91.12), ("Chenga", "Barpeta", 26.25, 91.05),
]

# Safeguard: a name+district must appear once. Duplicates broke the map
# counts once (chip said 10 RED, only 6 distinct dots) - never again.
_seen = set()
_unique = []
for _v in VILLAGES:
    _k = (_v[0], _v[1])
    if _k in _seen:
        print(f"  CONFIG WARNING: duplicate village {_k} dropped")
        continue
    _seen.add(_k)
    _unique.append(_v)
VILLAGES = _unique


def overpass_query(bbox: tuple) -> list[dict]:
    s, w, n, e = bbox
    query = (f'[out:json][timeout:90];'
             f'node["place"~"village|town|hamlet|suburb"]["name"]({s},{w},{n},{e});'
             f'out body;')
    last_err = None
    for attempt in range(3):
        resp = requests.get(OVERPASS_URL, params={"data": query}, timeout=120,
                            headers={"User-Agent": "JalDrishti-PoC/0.1 (flood feasibility demo)"})
        if resp.status_code in (429, 504):  # rate-limited / overloaded: wait and retry
            last_err = f"HTTP {resp.status_code}"
            time.sleep(20 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json().get("elements", [])
    raise RuntimeError(f"Overpass failed after retries: {last_err}")


def overpass_places(district: str, bbox: tuple) -> list[dict]:
    """All named place nodes in the district bbox; splits the bbox in half
    if the full query keeps timing out server-side."""
    try:
        elements = overpass_query(bbox)
    except RuntimeError:
        s, w, n, e = bbox
        mid = (w + e) / 2
        print(f"  Overpass {district}: full bbox failed, retrying as two halves")
        elements = overpass_query((s, w, n, mid))
        time.sleep(5)
        elements += overpass_query((s, mid, n, e))
    save_json(RAW_DIR / f"overpass_{district.lower()}.json", {"elements": elements})
    return elements


def main() -> int:
    # Villages are static places: if the existing payload is fresh (< 7 days)
    # and matches the config size, keep it instead of hammering Overpass -
    # which rate-limits shared cloud IPs hard (a 3-hourly refetch of
    # never-changing coordinates was always wasteful). --force refetches.
    out_path = DATA_DIR / "villages.json"
    if "--force" not in sys.argv and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            age_d = (time.time() - out_path.stat().st_mtime) / 86400
            if age_d < 7 and len(prev.get("villages", [])) == len(VILLAGES):
                print(f"OK villages: payload {age_d:.1f} d old, {len(VILLAGES)} villages - kept (--force to refetch)")
                return 0
        except (ValueError, OSError):
            pass
    osm_index: dict[str, list[dict]] = {}
    overpass_ok = {}
    for district, bbox in DISTRICT_BBOX.items():
        try:
            elements = overpass_places(district, bbox)
            overpass_ok[district] = True
            for el in elements:
                name = norm(el.get("tags", {}).get("name", ""))
                if name:
                    osm_index.setdefault(f"{district}:{name}", []).append(el)
            print(f"  Overpass {district}: {len(elements)} place nodes")
        except Exception as err:  # noqa: BLE001 - fallback rule applies
            overpass_ok[district] = False
            print(f"  Overpass {district} FAILED ({err}); using approximate coordinates")
        time.sleep(5)  # be polite: the public endpoint allows ~2 concurrent slots

    def lookup(district: str, name: str) -> list[dict]:
        key = norm(name)
        for cand in [key, *ALIASES.get(key, [])]:
            hits = osm_index.get(f"{district}:{cand}", [])
            if hits:
                return hits
        # prefix match, e.g. OSM "Gaurisagar Tiniali" for our "Gaurisagar"
        pref = []
        for k, els in osm_index.items():
            d, osm_name = k.split(":", 1)
            if d == district and osm_name.startswith(key + " "):
                pref.extend(els)
        return pref

    retrieved_at = utc_now_iso()
    out, n_osm = [], 0
    for name, district, flat, flon in VILLAGES:
        hits = lookup(district, name)
        if hits:
            # nearest OSM match to our prior, in case of duplicate names in bbox
            el = min(hits, key=lambda e: (e["lat"] - flat) ** 2 + (e["lon"] - flon) ** 2)
            out.append({
                "name": name, "district": district,
                "lat": el["lat"], "lon": el["lon"],
                "source": "OpenStreetMap via Overpass API (place node)",
                "coordinate_precision": "osm_node",
                "osm_id": el["id"],
                "retrieved_at": retrieved_at,
            })
            n_osm += 1
        else:
            out.append({
                "name": name, "district": district,
                "lat": flat, "lon": flon,
                "source": "estimated",
                "coordinate_precision": "approximate",
                "retrieved_at": retrieved_at,
            })

    # coordinate-collision guard: if OSM matched two different villages to the
    # same node (e.g. Barpeta vs Barpeta Road), keep the first and revert the
    # others to their own config coordinate so every village is a distinct dot.
    seen_xy: dict = {}
    cfg = {(n, d): (flat, flon) for n, d, flat, flon in VILLAGES}
    for v in out:
        key = (round(v["lat"], 5), round(v["lon"], 5))
        if key in seen_xy:
            flat, flon = cfg[(v["name"], v["district"])]
            v["lat"], v["lon"] = flat, flon
            v["source"] = "estimated (OSM node collided with another village)"
            v["coordinate_precision"] = "approximate"
            v.pop("osm_id", None)
            print(f"  collision: {v['name']} shared a node with "
                  f"{seen_xy[key]}; reverted to config coordinate")
        else:
            seen_xy[key] = v["name"]

    doc = {
        "generated_at": retrieved_at,
        "note": ("DEMO SUBSET (~80 of thousands). Village/settlement NAMES are real places in "
                 "Sivasagar, Morigaon, Cachar, Sonitpur, Golaghat, Charaideo, Nagaon and "
                 "Barpeta districts, Assam. Coordinates are OSM place "
                 "nodes where matched, otherwise approximate estimates (+/- a few km) - NOT "
                 "surveyed revenue-village boundaries. Points only, no polygons. No LGD codes "
                 "are claimed. Some entries are towns or mauza centres rather than strictly "
                 "revenue villages."),
        "overpass_attempted": True,
        "overpass_succeeded": overpass_ok,
        "osm_matched": n_osm,
        "villages": out,
    }
    save_json(DATA_DIR / "villages.json", doc)

    # ---- verification ----
    check = __import__("json").load(open(DATA_DIR / "villages.json", encoding="utf-8"))
    vs = check["villages"]
    assert len(vs) >= 45, f"need >= 45 villages, got {len(vs)}"
    assert {"Sivasagar", "Morigaon", "Cachar"} <= {v["district"] for v in vs}
    for v in vs:
        assert {"name", "district", "lat", "lon", "source", "coordinate_precision"} <= set(v)
    print(f"OK data/villages.json: {len(vs)} villages, {n_osm} with OSM coords, "
          f"{len(vs) - n_osm} approximate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
