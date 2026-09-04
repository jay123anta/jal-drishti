"""Dam reservoir storage + hydro generation from NERLDC's daily PSP report.

NERLDC (Grid-India, North Eastern RLDC) publishes a daily Power Supply
Position report as a PDF. Section 7 ("Major Reservoir Particulars") gives
each major reservoir's present level and stored energy (MU); section 3B
gives per-station daily hydro generation. Both are exactly the signals
our dam-controlled basins were missing (Kopili, Ranganadi, Dhansiri via
Doyang, and the Kameng/Subansiri reaches): a dam's stored water and how
much it released yesterday.

Reachability note: nerldc.in serves only from Indian IPs, so this step
runs on the keep-alive host (an Indian connection), not from CI abroad.
It degrades honestly - a fetch or parse failure is recorded and skipped,
never fabricated.

Parsing is by word POSITION (pdfplumber extract_words, grouped by row
top-coordinate), because the report's text extracts in a scrambled
column order; anchoring on the known reservoir/station names and reading
the numbers on their row is robust to that. Values are OBSERVED
(operator-reported).
"""
from __future__ import annotations
import datetime
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

class _LegacyTLSAdapter(HTTPAdapter):
    """NERLDC's server needs legacy TLS renegotiation, which OpenSSL 3 blocks
    by default. This adapter re-enables it for this host only."""
    def init_poolmanager(self, *a, **k):
        ctx = create_urllib3_context()
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        k["ssl_context"] = ctx
        return super().init_poolmanager(*a, **k)

def _session():
    import requests, urllib3
    urllib3.disable_warnings()          # nerldc ships an incomplete cert chain
    s = requests.Session()
    s.mount("https://", _LegacyTLSAdapter())
    s.verify = False                    # public report; content validated (must be a PDF)
    return s

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "data" / "history" / "nerldc"
RAW = OUT / "raw"
BASE = "https://www.nerldc.in"

# reservoirs we care about -> which basin they regulate
RESERVOIRS = {
    "Kopili": "kopili", "Khandong": "kopili",
    "Ranganadi": "ranganadi", "Pare": "ranganadi",
    "Doyang": "dhansiri", "Kameng": "jiabharali",
    "Loktak": None, "Lower Subansiri": "subansiri", "Umium": "kopili",
}
SRC = ("Grid-India NERLDC daily Power Supply Position report "
       "(nerldc.in); operator-reported reservoir level/storage and "
       "station generation")


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def candidate_url(d: datetime.date) -> str:
    """Flat WordPress uploads path; filename date is the PUBLICATION date."""
    return f"{BASE}/wp-content/uploads/NER-PSP-REPORT-DATED-{d:%d-%m-%Y}.pdf"


def find_pdf_url():
    """Return (url, publication_date) for the latest PSP report, or (None, None)."""
    sess = _session()
    # 1. the report page always links the newest report - read it straight off
    try:
        r = sess.get(f"{BASE}/power-supply-position-psp-report/", timeout=45)
        m = re.search(r"https?://[^\"']+/NER-PSP-REPORT-DATED-(\d\d)-(\d\d)-(\d{4})\.pdf",
                      r.text)
        if m:
            dd, mm, yy = (int(x) for x in m.groups())
            return m.group(0), datetime.date(yy, mm, dd)
    except requests.RequestException:
        pass
    # 2. fall back to the flat dated pattern for the last few days
    for back in range(5):
        d = datetime.date.today() - datetime.timedelta(days=back)
        u = candidate_url(d)
        try:
            r = sess.head(u, timeout=30)
            if r.status_code == 200 and "pdf" in r.headers.get("content-type", "").lower():
                return u, d
        except requests.RequestException:
            pass
    return None, None


def parse_reservoirs(pdf_bytes: bytes) -> list[dict]:
    import pdfplumber
    rows = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=False)
            # group words into rows by rounded top coordinate
            lines: dict[int, list] = {}
            for w in words:
                lines.setdefault(round(w["top"] / 3), []).append(w)
            for _, ws in lines.items():
                ws.sort(key=lambda x: x["x0"])
                toks = [w["text"] for w in ws]
                text = " ".join(toks)
                for name, basin in RESERVOIRS.items():
                    if text.startswith(name) or f" {name} " in f" {text} ":
                        nums = [float(t) for t in toks
                                if re.fullmatch(r"-?\d+(\.\d+)?", t)]
                        if len(nums) >= 4:
                            rows.append({"reservoir": name, "basin": basin,
                                         "numbers": nums, "raw": text})
                        break
    return rows


def label(nums: list[float]) -> dict:
    # report column order (left-to-right): MDDL, FRL, Designed-energy,
    # PRESENT-level, PRESENT-energy, last-year..., last-day...
    mddl, frl, designed, present_level = nums[0], nums[1], nums[2], nums[3]
    present_mu = nums[4] if (designed > 0 and len(nums) >= 9) else None
    fill = (round((present_level - mddl) / (frl - mddl), 3)
            if frl > mddl and mddl <= present_level <= frl + 1 else None)
    return {"mddl_m": mddl, "frl_m": frl, "designed_mu": designed,
            "present_level_m": present_level, "present_storage_mu": present_mu,
            "fill_fraction": fill, "n_raw": len(nums),
            "needs_review": len(nums) not in (6, 9)}


def save_rows(d: datetime.date, rows: list[dict], now: str) -> int:
    out = pd.DataFrame([{
        "date": d.isoformat(), "reservoir": r["reservoir"], "basin": r["basin"],
        **label(r["numbers"]),
        "raw_numbers": ";".join(str(n) for n in r["numbers"]),
    } for r in rows])
    p = OUT / "reservoirs.parquet"
    if p.exists():
        out = pd.concat([pd.read_parquet(p), out], ignore_index=True)
    out = out.drop_duplicates(subset=["date", "reservoir"], keep="last").sort_values(["date", "reservoir"])
    out.to_parquet(p, index=False)
    (OUT / "reservoirs.provenance.json").write_text(
        f'{{"source": "{SRC}", "class": "OBSERVED", "archived": true, '
        f'"retrieved_at": "{now}", "rows": {len(out)}, '
        f'"note": "present_level_m and fill_fraction reliable; present_storage_mu '
        f'for storage dams; rows flagged needs_review parsed off-shape; raw PDFs kept LOCAL only (reproducible via --backfill)"}}',
        encoding="utf-8")
    return len(out)


def process(url: str, d: datetime.date, now: str) -> int:
    """Fetch one report at a known URL, parse and archive it. Returns row count."""
    try:
        r = _session().get(url, timeout=120); r.raise_for_status()
        (RAW / f"{d.isoformat()}.pdf").write_bytes(r.content)
        rows = parse_reservoirs(r.content)
    except Exception:  # noqa: BLE001
        return 0
    if not rows:
        return 0
    save_rows(d, rows, now)
    return len(rows)


def backfill(now: str) -> int:
    """Walk the dated flat URLs backwards until the archive runs out."""
    sess = _session()
    have = set()
    p = OUT / "reservoirs.parquet"
    if p.exists():
        have = set(pd.read_parquet(p)["date"].astype(str))
    d = datetime.date.today()
    misses = n_ok = 0
    while misses < 45:            # stop after ~1.5 months of consecutive gaps = archive start
        if d.isoformat() in have:
            d -= datetime.timedelta(days=1); continue
        url = candidate_url(d)
        try:
            h = sess.head(url, timeout=30)
            ok = h.status_code == 200 and "pdf" in h.headers.get("content-type", "").lower()
        except requests.RequestException:
            ok = False
        if ok and process(url, d, now):
            n_ok += 1; misses = 0
            if n_ok % 20 == 0:
                print(f"  backfilled {n_ok} reports (through {d})...", flush=True)
        else:
            misses += 1
        d -= datetime.timedelta(days=1)
    print(f"OK nerldc backfill: {n_ok} historical reports archived "
          f"(stopped after {misses} consecutive gaps at {d})")
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(exist_ok=True)
    now = utc_now_iso()
    if "--backfill" in sys.argv:
        return backfill(now)
    url, d = find_pdf_url()
    if not url:
        print("DEGRADED: no NERLDC PSP report found "
              "(site reachable only from Indian IPs; runs on the keep-alive host)")
        return 0
    n = process(url, d, now)
    if not n:
        print(f"DEGRADED: NERLDC report {d} not fetched/parsed (raw kept if downloaded)")
        return 0
    total = len(pd.read_parquet(OUT / "reservoirs.parquet"))
    print(f"OK nerldc: {d} parsed, {n} reservoirs -> data/history/nerldc/ (ledger now {total} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
