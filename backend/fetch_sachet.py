"""Official flood alerts from NDMA's SACHET public CAP feed.

SACHET (sachet.ndma.gov.in) is the National Disaster Management
Authority's Common Alerting Protocol dissemination system. Its RSS/CAP
feed is machine-readable and marked "public domain" in the feed itself.
This is a standards-based public alert feed built for redistribution -
not portal scraping.

What this does, every run (>= 3-hourly, like the CWC AFF feed):
- fetch the all-India RSS, keep Assam-relevant items (sender Assam-SDMA
  or text mentioning Assam / our rivers);
- fetch each NEW alert's full CAP XML (raw archived gzip, one file per
  alert identifier - alerts vanish from the feed as they expire, so this
  archive is the lasting record);
- maintain data/history/sachet/alerts.parquet (deduped by identifier);
- publish public/sachet_alerts.json with the currently ACTIVE alerts for
  the viewer, every value provenance-wrapped.

Never fabricates: on any failure the payload records the real error and
keeps the previous alert list marked stale.
"""
from __future__ import annotations
import datetime
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
HIST = BASE_DIR / "data" / "history" / "sachet"
RAW = HIST / "raw"
PARQUET = HIST / "alerts.parquet"
STATE = HIST / "FETCH-STATE.json"
PAYLOAD = BASE_DIR / "public" / "sachet_alerts.json"

RSS_URL = "https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml"
XML_URL = "https://sachet.ndma.gov.in/cap_public_website/FetchXMLFile?identifier={ident}"
MIN_INTERVAL_H = 3
SRC = ("NDMA SACHET Common Alerting Protocol public feed "
       "(sachet.ndma.gov.in, feed marked public domain); alert content "
       "authored by the issuing agency (e.g. ASDMA, CWC, IMD)")

ASSAM_RE = re.compile(
    r"assam|guwahati|brahmaputra|barak|silchar|dibrugarh|tezpur|dhansiri|"
    r"jia.?bharali|kopili|beki|manas\b|subansiri|disang|dikhow|golaghat|"
    r"numaligarh|sivasagar|barpeta|lakhimpur|hailakandi|karimganj|cachar|sankosh|dhubri|golokganj",
    re.I)

# rivers this map covers (matched loosely against alert wording)
COVERED_RIVERS = ["BRAHMAPUTRA", "BARAK", "DIKHOW", "KOPILI", "BHARALI", "BEKI",
                  "DHANSIRI", "DISANG", "SUBANSIRI", "MANAS", "RANGANADI", "KATAKHAL", "SANKOSH"]

def river_in_alert(headline: str) -> str | None:
    m = re.search(r"River\s+([A-Za-z()\- ]+?)\s+at\s", headline)
    return m.group(1).strip() if m else None

CAP_FIELDS = ["identifier", "sender", "sent", "status", "msgType", "event",
              "urgency", "severity", "certainty", "effective", "onset",
              "expires", "headline", "description", "areaDesc"]


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wrap(v, cls="OBSERVED"):
    return {"value": v, "class": cls, "source": SRC, "retrieved_at": utc_now_iso()}


def cap_text(xml: str, tag: str) -> str:
    m = re.search(rf"<cap:{tag}>(.*?)</cap:{tag}>", xml, re.S)
    return (m.group(1).strip() if m else "")


def parse_alert(ident: str, xml: str) -> dict:
    row = {f: cap_text(xml, f) for f in CAP_FIELDS}
    row["identifier"] = row["identifier"] or ident
    # SACHET quirk (verified 2026-09-03): the CAP <altitude>/<ceiling>
    # fields carry the station's LAT/LON, not altitudes. Stored under
    # honest names; treated as approximate.
    lat, lon = cap_text(xml, "altitude"), cap_text(xml, "ceiling")
    try:
        row["approx_lat"], row["approx_lon"] = float(lat), float(lon)
    except ValueError:
        row["approx_lat"] = row["approx_lon"] = None
    row["archived_at"] = utc_now_iso()
    return row


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def too_soon(state: dict) -> bool:
    try:
        last = datetime.datetime.fromisoformat(state["last_fetch_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return False
    age_h = (datetime.datetime.now(datetime.timezone.utc) - last).total_seconds() / 3600
    return age_h < MIN_INTERVAL_H and "--force" not in sys.argv


def write_payload(active: list[dict], now: str, error: str | None, n_total: int,
                  uncovered: list | None = None) -> None:
    doc = {
        "generated_at": now, "source": SRC,
        "attribution": ("Official public alerts via NDMA SACHET; issued by the "
                        "responsible agency named in each alert (ASDMA / CWC / IMD)"),
        "fetch_policy": f"at most every {MIN_INTERVAL_H} h (public feed, standards-based)",
        "error": error, "alerts_archived_total": n_total,
        "uncovered_alert_rivers": uncovered or [],
        "active_alerts": [{
            "identifier": a["identifier"],
            "sender": wrap(a["sender"]),
            "event": wrap(a["event"]),
            "severity": wrap(a["severity"]),
            "headline": wrap(a["headline"]),
            "description": wrap(a["description"]),
            "area": wrap(a["areaDesc"]),
            "sent": wrap(a["sent"]),
            "expires": wrap(a["expires"]),
        } for a in active],
    }
    PAYLOAD.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def main() -> int:
    now = utc_now_iso()
    HIST.mkdir(parents=True, exist_ok=True); RAW.mkdir(exist_ok=True)
    state = load_state()
    if too_soon(state):
        print(f"OK sachet: last fetch < {MIN_INTERVAL_H} h ago - kept (--force to override)")
        return 0

    old = pd.read_parquet(PARQUET) if PARQUET.exists() else pd.DataFrame()
    known = set(old["identifier"]) if len(old) else set()

    try:
        rss = requests.get(RSS_URL, timeout=60)
        rss.raise_for_status()
        idents = re.findall(r"FetchXMLFile\?identifier=(\d+)", rss.text)
        items = re.findall(r"<item>(.*?)</item>", rss.text, re.S)
    except Exception as err:  # noqa: BLE001 - degrade honestly
        print(f"  sachet RSS FAILED: {err}")
        write_payload([], now, f"RSS fetch failed: {err}",
                      int(len(old)) if len(old) else 0, None)
        return 0

    # Assam-relevant identifiers (match on the RSS item text)
    wanted = []
    for item in items:
        m = re.search(r"FetchXMLFile\?identifier=(\d+)", item)
        if m and ASSAM_RE.search(item):
            wanted.append(m.group(1))
    print(f"  feed: {len(idents)} alerts, {len(wanted)} Assam-relevant")

    new_rows, n_fail = [], 0
    for ident in wanted:
        full_id = None
        # raw archive is per feed-identifier
        raw_p = RAW / f"{ident}.xml.gz"
        if raw_p.exists() and any(str(ident) in k for k in known):
            continue
        try:
            r = requests.get(XML_URL.format(ident=ident), timeout=60)
            r.raise_for_status()
            xml = r.text
            raw_p.write_bytes(gzip.compress(xml.encode("utf-8"), 6))
            row = parse_alert(ident, xml)
            full_id = row["identifier"]
            if full_id not in known:
                new_rows.append(row)
        except Exception as err:  # noqa: BLE001
            n_fail += 1
            print(f"  alert {ident}: FAILED ({err})")

    df = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else old
    if len(df):
        df = df.drop_duplicates(subset="identifier", keep="first")
        df.to_parquet(PARQUET, index=False)
        (HIST / "alerts.provenance.json").write_text(json.dumps({
            "source": SRC, "class": "OBSERVED", "archived": True,
            "retrieved_at": now, "rows": int(len(df)),
            "note": ("official alert messages; approx_lat/approx_lon come from "
                     "CAP altitude/ceiling fields (SACHET quirk) - approximate")},
            indent=1), encoding="utf-8")

    # active = not yet expired (expires is IST like +05:30 offsets in feed)
    active = []
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    for _, a in (df.iterrows() if len(df) else []):
        try:
            exp = datetime.datetime.fromisoformat(a["expires"])
            if exp.tzinfo is None:
                continue
            if exp > now_dt:
                active.append(a.to_dict())
        except (ValueError, TypeError):
            continue
    active.sort(key=lambda a: a.get("sent", ""), reverse=True)

    # coverage watch: alert rivers this map does not cover yet
    uncovered = sorted({rv for a in active
                        for rv in [river_in_alert(str(a.get("headline", "")))]
                        if rv and not any(k in rv.upper() for k in COVERED_RIVERS)})
    if uncovered:
        print(f"  COVERAGE GAP: active alerts mention uncovered river(s): {', '.join(uncovered)}")
        (HIST / "coverage_gaps.json").write_text(json.dumps(
            {"at": now, "uncovered_rivers": uncovered}, indent=1), encoding="utf-8")
    write_payload(active, now, None, int(len(df)), uncovered)
    state.update({"last_fetch_at": now})
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"OK sachet: {len(new_rows)} new alerts archived ({n_fail} failed), "
          f"{len(df)} total, {len(active)} active -> public/sachet_alerts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
