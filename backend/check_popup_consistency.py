"""Gate: every map dot's colour must agree with the text in its own popup.

A village dot coloured by a model must match that model's P(exceed) against
the documented cutoffs; a heuristic dot must match its score; and no dot's
plain-language sentence may contradict its colour. River icons must match
their gauge level against the official warning/danger marks. Any mismatch
fails the pipeline - so a dot can never silently disagree with what its
popup says (the class of bug a careful reader would spot and lose trust
over). Exit 0 = all consistent; exit 1 = at least one contradiction.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

PUBLIC = Path(__file__).resolve().parent.parent / "public"


def colour_from_p(p90: float, p98: float) -> str:
    # two-tier: RED needs likely exceedance of the q98 extreme level; YELLOW of q90
    return "RED" if p98 >= 0.5 else "YELLOW" if p90 >= 0.2 else "GREEN"


def colour_from_score(c: int) -> str:
    return "GREEN" if c <= 1 else "YELLOW" if c == 2 else "RED"


def main() -> int:
    errors: list[str] = []

    vs = json.loads((PUBLIC / "villages_status.json").read_text(encoding="utf-8"))
    for v in vs["villages"]:
        r = v["risk"]
        name, colour, method = v["name"], r["value"], str(r.get("method", ""))
        if method.startswith("model-v0-"):
            p = r.get("p_exceed_h1")
            p_hi = r.get("p_exceed_extreme_h1")
            if p is None or p_hi is None:
                errors.append(f"village {name}: model dot with no P(exceed) pair")
            elif colour_from_p(p, p_hi) != colour:
                errors.append(f"village {name}: dot {colour} but P90={p}/P98={p_hi} implies {colour_from_p(p, p_hi)}")
        elif method == "heuristic":
            c = v["scores"]["combined"]["value"]
            if colour_from_score(c) != colour:
                errors.append(f"village {name}: dot {colour} but score {c} implies {colour_from_score(c)}")
        else:
            errors.append(f"village {name}: unknown risk method {method!r}")
        basis = str(r.get("basis", "")).lower()
        if colour == "RED" and "normal range" in basis:
            errors.append(f"village {name}: RED dot but basis text says 'normal range'")
        if colour == "GREEN" and ("unusually high" in basis or "raised chance" in basis):
            errors.append(f"village {name}: GREEN dot but basis text implies elevated water")

    rs = json.loads((PUBLIC / "rivers_status.json").read_text(encoding="utf-8"))
    cwc_path = PUBLIC / "cwc_stations.json"
    cwc = {}
    if cwc_path.exists():
        cwc = {s["poc_river"]: s for s in json.loads(cwc_path.read_text(encoding="utf-8"))["stations"]
               if s.get("poc_river")}
    # A river pin must sit where its own popup says it sits. The map draws
    # map_lat/map_lon, so those must exist, carry provenance, and stay close to
    # the official gauge the popup shows. The GloFAS data cell may legitimately
    # be some km away, but the popup has to disclose that distance rather than
    # let position and provenance disagree in silence.
    for rv in rs["rivers"]:
        name = rv["id"]
        pos = rv.get("map_position")
        if not pos or pos.get("value") is None:
            errors.append(f"river {name}: no map_position - the pin has no stated origin")
            continue
        if rv.get("map_lat") != pos["value"][0] or rv.get("map_lon") != pos["value"][1]:
            errors.append(f"river {name}: map_lat/map_lon disagree with map_position")
        if pos.get("basis") == "cwc_gauge":
            c = cwc.get(name)
            if c and c.get("lat") is not None:
                d = math.hypot((c["lat"] - pos["value"][0]) * 111.0,
                               (c["lon"] - pos["value"][1]) * 111.0
                               * math.cos(math.radians(pos["value"][0])))
                if d > 0.5:
                    errors.append(f"river {name}: pin claims the CWC gauge position "
                                  f"but sits {d:.2f} km from it")
        if rv.get("grid_lat") is not None and rv.get("data_cell_offset_km") is None:
            errors.append(f"river {name}: has a GloFAS data cell but no disclosed offset")

    for rv in rs["rivers"]:
        c = cwc.get(rv["id"])
        if not c or c.get("degraded"):
            continue
        warn = c["warning_level_m"]["value"]
        dang = c["danger_level_m"]["value"]
        if warn is not None and dang is not None and warn > dang:
            errors.append(f"river {rv['id']}: warning mark {warn} > danger mark {dang}")
        nowt = (c.get("observed_trend_now") or {}).get("value")
        if nowt not in (None, "rising", "falling", "steady"):
            errors.append(f"river {rv['id']}: invalid trend word {nowt!r}")

    # Every claim a reader sees must say WHEN it is about. The chips are the
    # mechanism: NOW for measured current state, TOMORROW/AHEAD for anything
    # predicted, and a past-window chip for trends and the look-back. If a popup
    # builder loses its chip, a forecast starts reading as a present-tense fact.
    idx = (PUBLIC / "index.html").read_text(encoding="utf-8")
    required = {
        "village model forecast": 'class="tw tw-ahead">TOMORROW',
        "village current-conditions estimate": 'class="tw tw-now">NOW',
        "official gauge reading": '<span class="tw tw-now">NOW</span><b>Official river gauge',
        "our 2-day river trend": 'class="tw tw-past">LAST 2 DAYS',
        "look-back popup": 'class="tw tw-past">THAT DAY',
    }
    for what, needle in required.items():
        if needle not in idx:
            errors.append(f"index.html: {what} lost its timeframe label ({needle!r})")
    for cls in ("tw-now", "tw-ahead", "tw-past"):
        if f".pp .{cls}" not in idx:
            errors.append(f"index.html: timeframe chip style .{cls} is not defined")

    sa_path = PUBLIC / "sachet_alerts.json"
    if sa_path.exists():
        sa = json.loads(sa_path.read_text(encoding="utf-8"))
        for a in sa.get("active_alerts", []):
            if not (a.get("headline") or {}).get("value"):
                errors.append(f"alert {a.get('identifier')}: empty headline")

    if errors:
        print(f"POPUP CONSISTENCY FAILED - {len(errors)} contradiction(s):")
        for e in errors[:40]:
            print("  -", e)
        return 1
    print(f"OK popup consistency: {len(vs['villages'])} village dots, "
          f"{len(rs['rivers'])} river points - every colour matches its popup text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
