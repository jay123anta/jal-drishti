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


def candidate_urls(d: datetime.date) -> list[str]:
    """WordPress uploads path with the dated filename pattern we observed."""
    fn = f"NER-PSP-REPORT-DATED-{d:%d-%m-%Y}.pdf"
    return [
        f"{BASE}/wp-content/uploads/{d:%Y/%m}/{fn}",
        f"{BASE}/wp-content/uploads/{d:%Y}/{d:%m}/{fn}",
    ]


def find_pdf_url(d: datetime.date) -> str | None:
    # 1. try the known dated patterns directly
    for u in candidate_urls(d):
        try:
            r = _session().head(u, timeout=30)
            if r.status_code == 200 and "pdf" in r.headers.get("content-type", "").lower():
                return u
        except requests.RequestException:
            pass
    # 2. fall back to scraping the PSP page for a link to the dated file
    try:
        r = _session().get(f"{BASE}/power-supply-position-psp-report/", timeout=45)
        m = re.search(r'https?://[^"\']+NER-PSP-REPORT-DATED-'
                      + f"{d:%d-%m-%Y}" + r'\.pdf', r.text)
        if m:
            return m.group(0)
    except requests.RequestException:
        pass
    return None


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(exist_ok=True)
    now = utc_now_iso()
    # the report is published for the previous day
    d = datetime.date.today() - datetime.timedelta(days=1)
    url = find_pdf_url(d)
    if not url:
        # try the day before as well (weekend/holiday lag)
        d = d - datetime.timedelta(days=1)
        url = find_pdf_url(d)
    if not url:
        print(f"DEGRADED: no NERLDC PSP report found for {d} "
              "(site reachable only from Indian IPs; runs on the keep-alive host)")
        return 0
    try:
        r = _session().get(url, timeout=120); r.raise_for_status()
        (RAW / f"{d.isoformat()}.pdf").write_bytes(r.content)
        rows = parse_reservoirs(r.content)
    except Exception as err:  # noqa: BLE001
        print(f"DEGRADED: NERLDC fetch/parse failed for {d}: {err}")
        return 0
    if not rows:
        print(f"DEGRADED: NERLDC report {d} fetched but no reservoir rows parsed "
              "(layout may have changed - raw PDF kept for inspection)")
        return 0
    out = pd.DataFrame([{
        "date": d.isoformat(), "reservoir": r["reservoir"], "basin": r["basin"],
        "raw_numbers": ";".join(str(n) for n in r["numbers"]), "raw_row": r["raw"],
    } for r in rows])
    p = OUT / "reservoirs.parquet"
    if p.exists():
        out = pd.concat([pd.read_parquet(p), out], ignore_index=True)
    out = out.drop_duplicates(subset=["date", "reservoir"], keep="last").sort_values(["date", "reservoir"])
    out.to_parquet(p, index=False)
    (OUT / "reservoirs.provenance.json").write_text(
        f'{{"source": "{SRC}", "class": "OBSERVED", "archived": true, '
        f'"retrieved_at": "{now}", "rows": {len(out)}, '
        f'"note": "raw_numbers are the report row values pending column mapping '
        f'confirmed against a real run; raw PDFs kept in raw/"}}', encoding="utf-8")
    print(f"OK nerldc: {d} parsed, {len(rows)} reservoirs -> data/history/nerldc/ "
          f"(ledger now {len(out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
