"""Gate: stop the run before publishing if most live inputs failed to arrive.

When a fetch fails, the fetchers fill in a clearly labelled SIMULATED series so
the rest of the pipeline can still run for testing. That is fine for a single
point on a bad afternoon - classify_risk then keeps those villages at their last
real-data estimate. It is not fine when the whole network is gone: on
2026-09-19 the laptop woke with no internet, every rain and river fetch failed,
and the fallback formula painted 97 of 149 villages red from placeholder numbers.

This gate runs straight after the fetches. If fewer than MIN_LIVE of the rain
points or of the river points are live, it exits 1: the pipeline stops, nothing
later in the run overwrites the map, and the keep-alive does not commit. The
published site keeps its last good data, whose own timestamp shows its age, and
the next scheduled or wake-up run tries again.

Official gauges (CWC) are reported but do not stop the run: the colours do not
depend on them, and a CWC outage alone leaves the rest of the map honest.
Exit 0 = enough live data; exit 1 = do not publish this run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PUBLIC = Path(__file__).resolve().parent.parent / "public"
MIN_LIVE = 0.5


def share(points: list[dict]) -> tuple[int, int]:
    return sum(1 for p in points if p.get("live")), len(points)


def main() -> int:
    try:
        rain = json.loads((DATA / "rainfall.json").read_text(encoding="utf-8"))["points"]
        river = json.loads((DATA / "discharge.json").read_text(encoding="utf-8"))["points"]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"LIVE INPUTS CHECK FAILED - cannot read this run's fetch results ({e})")
        return 1

    r_live, r_all = share(rain)
    q_live, q_all = share(river)
    try:
        cwc = json.loads((PUBLIC / "cwc_stations.json").read_text(encoding="utf-8"))["stations"]
        c_live = sum(1 for s in cwc if not s.get("degraded"))
        cwc_note = f", official gauges {c_live}/{len(cwc)}"
    except (FileNotFoundError, ValueError, KeyError):
        cwc_note = ""

    summary = f"rain {r_live}/{r_all} live, rivers {q_live}/{q_all} live{cwc_note}"
    low = [name for name, (n, d) in (("rain", (r_live, r_all)), ("rivers", (q_live, q_all)))
           if d == 0 or n / d < MIN_LIVE]
    if low:
        print(f"LIVE INPUTS CHECK FAILED - {summary}. Under {MIN_LIVE:.0%} live for "
              f"{' and '.join(low)}: most likely no internet, or a source is down. "
              "Stopping before anything is recoloured or published; the site keeps "
              "its last good data and the next run will try again.")
        return 1
    print(f"OK live inputs: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
