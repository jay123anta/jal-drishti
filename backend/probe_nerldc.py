"""One-off diagnostic: find how NERLDC serves its PSP PDF. Run on the laptop."""
import re
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

BASE = "https://www.nerldc.in"
PAGE = f"{BASE}/power-supply-position-psp-report/"

sess = _session()
r = sess.get(PAGE, timeout=45)
print("PAGE status:", r.status_code, "| length:", len(r.text))

# anything that looks like a PDF link or an uploads path or a PSP filename
hits = set()
for pat in (r'https?://[^\s"\'<>]+\.pdf',
            r'[\'"][^\s"\'<>]*(?:uploads|reports|psp|PSP)[^\s"\'<>]*[\'"]',
            r'NER-PSP-REPORT[^\s"\'<>]*',
            r'[\'"]/wp-[^\s"\'<>]+[\'"]',
            r'ajaxurl|admin-ajax|\.php[^\s"\'<>]*'):
    for m in re.findall(pat, r.text):
        hits.add(m.strip('\'"'))
print("\n--- candidate paths/links found in the page ---")
for h in sorted(hits)[:40]:
    print(" ", h)

# common WordPress listing of the uploads month folder
for path in ("/wp-content/uploads/2026/09/",
             "/wp-json/wp/v2/media?search=PSP&per_page=5"):
    try:
        rr = sess.get(BASE + path, timeout=30)
        print(f"\n{path} -> {rr.status_code}")
        found = re.findall(r'NER-PSP-REPORT-DATED-\d\d-\d\d-\d{4}\.pdf', rr.text)
        if found:
            print("  filenames:", sorted(set(found))[-5:])
        elif path.endswith("/"):
            print("  (directory listing not exposed)")
    except requests.RequestException as e:
        print(f"\n{path} -> error {e}")
