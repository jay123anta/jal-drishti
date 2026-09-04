"""UPGRADE v2, STEP E - plain-language gate for the viewer's DEFAULT view.

The map must be understandable by a novice with no explanation. This gate
fails (exit 1) if any banned technical word appears in the default-view
text of public/index.html, or in the plain-language section of
public/replay_findings.json (which feeds the default-view findings card).

What counts as "default view" in index.html (structural convention,
established in Step D):
- everything EXCEPT the <style> block,
- EXCEPT HTML wrapped in  <!-- TECH-START --> ... <!-- TECH-END -->,
- EXCEPT JS between      // TECH-STRINGS-START ... // TECH-STRINGS-END
  (those strings render only inside collapsed "Technical details" blocks
  or the Technical view panel),
- EXCEPT HTML/JS syntax: attribute assignments (class="...", href="...")
  and JS line comments, which are not user-visible text.

Banned words (case-insensitive, whole words) - per the Step E spec:
  unvalidated, classification, provenance, heuristic, simulated, payload,
  class, OBSERVED, FORECAST, reanalysis, percentile
(observed/forecast are checked in ANY case: the plain view should not use
them at all - it says "expected rain" instead.)
"""

import re
import sys

from common import PUBLIC_DIR, load_json

BANNED_CI = ["unvalidated", "classification", "provenance", "heuristic",
             "simulated", "payload", "class", "reanalysis", "percentile"]
# the class tokens are banned as the uppercase enum tokens they are; the
# plain English words remain allowed in the default view (Model v0 badges
# legitimately say "forecast: tested model") - a documented project rule).
BANNED_CS = ["OBSERVED", "FORECAST"]
BANNED = BANNED_CI + BANNED_CS
PATTERNS = ([(w, re.compile(rf"\b{w}\b", re.IGNORECASE)) for w in BANNED_CI]
            + [(w, re.compile(rf"\b{w}\b")) for w in BANNED_CS])


def default_view_lines(html: str):
    """Yield (lineno, text) for default-view content only."""
    in_style = in_tech_html = in_tech_js = False
    for i, raw in enumerate(html.splitlines(), 1):
        s = raw
        if "<style>" in s:
            in_style = True
            continue
        if "</style>" in s:
            in_style = False
            continue
        if "TECH-START" in s:
            in_tech_html = True
            continue
        if "TECH-END" in s:
            in_tech_html = False
            continue
        if "TECH-STRINGS-START" in s:
            in_tech_js = True
            continue
        if "TECH-STRINGS-END" in s:
            in_tech_js = False
            continue
        if in_style or in_tech_html or in_tech_js:
            continue
        s = re.sub(r"^\s*//.*", "", s)                 # JS line comments
        s = re.sub(r'[A-Za-z-]+="[^"]*"', " ", s)      # attr="..." (incl. class=)
        s = re.sub(r"[A-Za-z-]+='[^']*'", " ", s)
        yield i, s


def main() -> int:
    problems = []

    html_path = PUBLIC_DIR / "index.html"
    html = html_path.read_text(encoding="utf-8")
    for marker in ("TECH-START", "TECH-END", "TECH-STRINGS-START", "TECH-STRINGS-END"):
        if marker not in html:
            problems.append(f"index.html: structural marker {marker} missing "
                            f"(the gate cannot tell default view from technical view)")
    for lineno, text in default_view_lines(html):
        for word, pat in PATTERNS:
            if pat.search(text):
                problems.append(f"index.html:{lineno}: banned word '{word}' in "
                                f"default view: {text.strip()[:90]}")

    rf = load_json(PUBLIC_DIR / "replay_findings.json")
    for key, txt in (rf.get("plain") or {}).items():
        for word, pat in PATTERNS:
            if pat.search(str(txt)):
                problems.append(f"replay_findings.json plain.{key}: banned word "
                                f"'{word}': {str(txt)[:90]}")

    # model basis sentences render as the popup lead for model villages
    vs = load_json(PUBLIC_DIR / "villages_status.json")
    for v in vs.get("villages", []):
        basis = v.get("risk", {}).get("basis")
        if not basis:
            continue
        for word, pat in PATTERNS:
            if pat.search(basis):
                problems.append(f"villages_status {v['name']}: banned word "
                                f"'{word}' in model basis: {basis[:90]}")

    if problems:
        print(f"PLAIN-LANGUAGE CHECK FAILED - {len(problems)} problem(s):")
        for p in problems[:40]:
            print("  -", p)
        return 1
    n = sum(1 for _ in default_view_lines(html))
    print(f"OK plain language: no banned words in {n} default-view lines of "
          f"index.html or the findings card plain text "
          f"({', '.join(BANNED)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
