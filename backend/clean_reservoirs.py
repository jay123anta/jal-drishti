"""Clean the NERLDC reservoir ledger into a trustworthy per-dam feature.

The raw parse over 7 years is noisy (scrambled columns vary by report
layout, so MDDL/FRL are often misread and fill_fraction exceeds 1). The
present LEVEL, though, is the single most reliably-placed number. This
builds a robust feature from level alone:

  norm_level = clip((level - p2) / (p98 - p2), 0, 1)

where p2/p98 are each dam's own 2nd/98th percentile over its history -
so we normalise by what the dam actually does, never by a parsed design
value. Rows whose level falls far outside the dam's own plausible band
are dropped as parse errors (counted, never fabricated). Output:
data/history/nerldc/reservoirs_clean.parquet + a coverage report.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "history" / "nerldc"


def main() -> int:
    src = OUT / "reservoirs.parquet"
    if not src.exists():
        print("no reservoirs.parquet - run fetch_nerldc_psp.py --backfill first")
        return 0
    d = pd.read_parquet(src)
    d = d.dropna(subset=["present_level_m"])
    d["date"] = pd.to_datetime(d["date"])

    # per-year data-quality view, so the cutoff year is chosen on evidence
    d["year"] = d["date"].dt.year
    d["impossible_fill"] = d["fill_fraction"].fillna(0) > 1.05
    print("year-by-year quality (higher needs_review / impossible = worse layout):")
    q = d.groupby("year").agg(rows=("date", "size"),
                              needs_review=("needs_review", "mean"),
                              impossible=("impossible_fill", "mean"))
    for y, r in q.iterrows():
        print(f"  {y}: {int(r['rows']):5} rows | needs_review {r['needs_review']*100:4.0f}% "
              f"| impossible-fill {r['impossible']*100:4.0f}%")

    since = None
    for a in sys.argv:
        if a.startswith("--since="):
            since = int(a.split("=")[1])
    if since:
        d = d[d["year"] >= since]
        print(f"\n(restricting to {since} onward: {len(d)} rows)\n")

    clean = []
    print(f"{'reservoir':16} {'raw':>5} {'kept':>5} {'monsoon-days/yr':>16} {'norm range':>12}")
    for name, g in d.groupby("reservoir"):
        lv = g["present_level_m"].astype(float)
        p2, p98 = lv.quantile(0.02), lv.quantile(0.98)
        if p98 - p2 < 0.5:            # a dam whose level never moves is unusable
            print(f"{name:16} {len(g):>5}  (level essentially constant - skipped)")
            continue
        band = (lv >= p2 - (p98 - p2) * 0.5) & (lv <= p98 + (p98 - p2) * 0.5)
        gg = g[band].copy()
        gg["norm_level"] = ((gg["present_level_m"] - p2) / (p98 - p2)).clip(0, 1).round(3)
        gg["reservoir"] = name
        clean.append(gg[["date", "reservoir", "basin", "present_level_m", "norm_level"]])
        mon = gg[gg["date"].dt.month.isin([6, 7, 8, 9])]
        yrs = mon["date"].dt.year.nunique() or 1
        print(f"{name:16} {len(g):>5} {len(gg):>5} {len(mon)//yrs:>16} "
              f"{gg['norm_level'].min():.2f}-{gg['norm_level'].max():.2f}")
    if not clean:
        print("nothing clean enough - not writing")
        return 0
    out = pd.concat(clean, ignore_index=True).sort_values(["reservoir", "date"])
    out.to_parquet(OUT / "reservoirs_clean.parquet", index=False)
    (OUT / "reservoirs_clean.provenance.json").write_text(
        '{"source": "cleaned from NERLDC reservoir ledger - per-dam level '
        'normalised by own 2nd/98th percentile; MDDL/FRL parse columns not '
        'used", "class": "OBSERVED", "rows": %d}' % len(out), encoding="utf-8")
    print(f"\nOK cleaned: {len(out)} rows -> reservoirs_clean.parquet "
          f"({out['reservoir'].nunique()} dams)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
