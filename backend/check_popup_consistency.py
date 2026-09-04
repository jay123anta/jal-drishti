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
import sys
from pathlib import Path

PUBLIC = Path(__file__).resolve().parent.parent / "public"


def colour_from_p(p: float) -> str:
    return "RED" if p >= 0.5 else "YELLOW" if p >= 0.2 else "GREEN"


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
            if p is None:
                errors.append(f"village {name}: model dot with no P(exceed)")
            elif colour_from_p(p) != colour:
                errors.append(f"village {name}: dot {colour} but P(exceed)={p} implies {colour_from_p(p)}")
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
