"""Archive every model forecast as plain data - the prediction ledger.

Each pipeline run writes data/model_forecast_<basin>.json (tomorrow's
expected discharge, the probability of exceeding the seasonal threshold,
and the resulting colour). Those snapshots get overwritten every run, so
their history lived only in git commits - fragile, and awkward to score.

This step appends the current snapshot of every basin to
data/history/model_fc/<basin>.parquet (deduplicated on generated_at), so
the full ledger of what the system predicted, and when, is an ordinary
versioned dataset. The season scorecard reads these tables directly.
"""
from __future__ import annotations
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"
OUT = DATA / "history" / "model_fc"

FIELDS = ["generated_at", "basin", "degraded", "shipped", "q_now_m3s",
          "predicted_h1_m3s", "p_exceed_h1", "threshold_m3s", "colour", "error"]


def _val(node):
    return node.get("value") if isinstance(node, dict) else node


def row_from_payload(doc: dict) -> dict:
    preds = doc.get("predictions") or {}
    h1 = preds.get("h1") or {}
    inputs = doc.get("inputs") or {}
    return {
        "generated_at": doc.get("generated_at"),
        "basin": doc.get("basin"),
        "degraded": bool(doc.get("degraded")),
        "shipped": doc.get("shipped") or doc.get("model"),
        "q_now_m3s": _val(inputs.get("q_today_m3s")),
        "predicted_h1_m3s": _val(h1.get("q_m3s")),
        "p_exceed_h1": _val(h1.get("p_exceed_threshold")),
        "threshold_m3s": _val(doc.get("threshold_m3s")),
        "colour": _val(doc.get("colour")),
        "error": doc.get("error"),
    }


def append_rows(basin: str, rows: list[dict], retrieved_at: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{basin}.parquet"
    new = pd.DataFrame(rows, columns=FIELDS)
    if p.exists():
        old = pd.read_parquet(p)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset="generated_at", keep="first").sort_values("generated_at")
    new.to_parquet(p, index=False)
    (OUT / f"{basin}.provenance.json").write_text(json.dumps({
        "source": "this project's own model-forecast snapshots, archived per run",
        "class": "FORECAST", "archived": True, "retrieved_at": retrieved_at,
        "rows": int(len(new)),
        "first": str(new["generated_at"].min()), "last": str(new["generated_at"].max()),
        "note": ("the prediction ledger: every value the models published, with its "
                 "publication time - the season scorecard is computed from these")},
        indent=1), encoding="utf-8")
    return len(new)


def main() -> int:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_basins = total = 0
    for fp in sorted(DATA.glob("model_forecast_*.json")):
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except (ValueError, OSError) as err:
            print(f"  {fp.name}: unreadable ({err})")
            continue
        basin = doc.get("basin") or fp.stem.replace("model_forecast_", "")
        n = append_rows(basin, [row_from_payload(doc)], now)
        n_basins += 1
        total += n
    print(f"OK model forecast ledger: {n_basins} basins appended, "
          f"{total} total rows -> data/history/model_fc/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
