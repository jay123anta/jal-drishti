"""PHASE 1 (AI-ready platform) - the LABEL STORE: data/labels/events.csv

Without observed flood/level-crossing events there is no event-based
validation and no supervised flood model. This file holds every event the
project knows about, one row each, with its SOURCE and CONFIDENCE so that
low-confidence labels can be excluded from training.

Columns: event_id, basin, site, onset_utc, end_utc, peak_value, unit,
threshold, threshold_def, source, class, confidence, notes

Populated from:
1. GloFAS q90 crossings (monsoon threshold from models/dikhow_v0_meta.json)
   in the 2015-2026 history of the Dikhow target cell - modelled series,
   confidence "medium".
2. The real 19 July 2026 event (Mon cloudburst -> Upper Assam floods) from
   project context / REPLAY-FINDINGS.md - date from reporting, no gauge:
   confidence "medium".
3. CWC observed WARNING-level crossings appended automatically by
   fetch_cwc_aff.py from the growing gauge archive - confidence "high".

Run:  python backend/model/build_labels.py   (idempotent; keeps CWC rows)
"""

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import DATA_DIR, OBSERVED, REPO_ROOT, load_json, save_json, utc_now_iso  # noqa: E402

LABELS_DIR = DATA_DIR / "labels"
LABELS_CSV = LABELS_DIR / "events.csv"
COLUMNS = ["event_id", "basin", "site", "onset_utc", "end_utc", "peak_value", "unit",
           "threshold", "threshold_def", "source", "class", "confidence", "notes"]

SRC_GLOFAS = ("GloFAS v4 reanalysis via Open-Meteo Flood API (MODELLED product, "
              "not observed river data) - q90 crossing of the 2015-2025 monsoon "
              "distribution")
SRC_EVENT_2026 = ("Project context + REPLAY-FINDINGS.md: cloudburst in Mon district, "
                  "Nagaland on 19 July 2026 causing flooding in Sivasagar/Charaideo/"
                  "Jorhat/Golaghat (date from reporting; no gauge data)")
SRC_CWC = ("Central Water Commission Advisory Flood Forecast public dissemination "
           "portal - WIMS observed hourly level crossing the official WARNING level")


def load_labels() -> pd.DataFrame:
    if LABELS_CSV.exists():
        return pd.read_csv(LABELS_CSV, dtype=str).fillna("")
    return pd.DataFrame(columns=COLUMNS)


def save_labels(df: pd.DataFrame, retrieved_at: str) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    df = df.sort_values(["basin", "onset_utc"])[COLUMNS]
    df.to_csv(LABELS_CSV, index=False)
    save_json(LABELS_DIR / "events.provenance.json", {
        "file": "events.csv", "rows": int(len(df)), "columns": COLUMNS,
        "class": OBSERVED, "source": "see per-row source column",
        "retrieved_at": retrieved_at,
        "confidence_levels": {"high": "observed gauge crossing (CWC)",
                              "medium": "modelled-series crossing or reported date",
                              "low": "not used for training"},
        "by_source": {k: int(v) for k, v in df["source"].str[:40].value_counts().items()},
    })


def runs_above(series: pd.Series, thr: float) -> list[tuple]:
    """Contiguous runs of series >= thr -> [(onset, end, peak)] on the index."""
    out, start, peak = [], None, None
    prev_idx = None
    for idx, v in series.items():
        if v >= thr:
            if start is None:
                start, peak = idx, v
            else:
                peak = max(peak, v)
            prev_idx = idx
        elif start is not None:
            out.append((start, prev_idx, peak))
            start, peak = None, None
    if start is not None:
        out.append((start, prev_idx, peak))
    return out


def glofas_events(thr: float) -> list[dict]:
    d = DATA_DIR / "history" / "discharge" / "dikhow_sivasagar"
    parts = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))]
    if not parts:
        return []
    df = pd.concat(parts)
    q = pd.Series(df["discharge_m3s"].to_numpy(), index=pd.to_datetime(df["date"])).sort_index()
    rows = []
    for onset, end, peak in runs_above(q, thr):
        rows.append({
            "event_id": f"dikhow_glofas_q90_{onset.date()}",
            "basin": "dikhow", "site": "dikhow_sivasagar (GloFAS cell)",
            "onset_utc": onset.strftime("%Y-%m-%d"), "end_utc": end.strftime("%Y-%m-%d"),
            "peak_value": round(float(peak), 1), "unit": "m3/s",
            "threshold": thr, "threshold_def": "q90 of 2015-2025 monsoon GloFAS reanalysis",
            "source": SRC_GLOFAS, "class": OBSERVED, "confidence": "medium",
            "notes": "modelled series; daily resolution",
        })
    return rows


def event_2026() -> dict:
    return {
        "event_id": "upper_assam_2026-07-19", "basin": "dikhow",
        "site": "Sivasagar/Charaideo/Jorhat/Golaghat (multi-basin)",
        "onset_utc": "2026-07-19", "end_utc": "2026-07-22",
        "peak_value": "", "unit": "", "threshold": "", "threshold_def": "reported flood",
        "source": SRC_EVENT_2026, "class": OBSERVED, "confidence": "medium",
        "notes": "real disaster with loss of life; GloFAS cell peaked 766 m3/s on 2026-07-20",
    }


def cwc_crossings_to_labels(stations: list, obs_dir: pathlib.Path, table: dict,
                            retrieved_at: str) -> int:
    """Called by fetch_cwc_aff.py after each fetch: append observed WARNING
    crossings from the hourly archive. Returns number of new rows."""
    import re
    df = load_labels()
    known = set(df["event_id"])
    new = []
    for st in stations:
        meta = table.get(st["aff"]) or {}
        warn = meta.get("warning_m")
        pq = obs_dir / (re.sub(r"[^A-Za-z0-9]+", "_", st["aff"]).strip("_") + ".parquet")
        if warn is None or not pq.exists():
            continue
        a = pd.read_parquet(pq).dropna(subset=["time_utc"])
        s = pd.Series(a["wims_level_m"].to_numpy(),
                      index=pd.to_datetime(a["time_utc"], utc=True)).sort_index()
        for onset, end, peak in runs_above(s, float(warn)):
            eid = f"cwc_{st['aff'].lower().replace(' ', '_')}_{onset.strftime('%Y-%m-%dT%H')}"
            if eid in known:
                continue
            new.append({
                "event_id": eid, "basin": st.get("poc_river") or st["aff"].lower(),
                "site": f"CWC {st['aff']} ({meta.get('river', '')})",
                "onset_utc": onset.strftime("%Y-%m-%dT%H:%MZ"),
                "end_utc": end.strftime("%Y-%m-%dT%H:%MZ"),
                "peak_value": round(float(peak), 2), "unit": "m",
                "threshold": warn, "threshold_def": "CWC official WARNING level",
                "source": SRC_CWC, "class": OBSERVED, "confidence": "high",
                "notes": f"danger level {meta.get('danger_m')} m; hourly observed",
            })
    if new:
        df = pd.concat([df, pd.DataFrame(new)], ignore_index=True)
        save_labels(df, retrieved_at)
    return len(new)


def main() -> int:
    now = utc_now_iso()
    thr = load_json(REPO_ROOT / "models" / "dikhow_v0_meta.json")["threshold_q90_monsoon_2015_2025_m3s"]
    df = load_labels()
    keep = df[df["source"].str.startswith("Central Water Commission")] if len(df) else df
    rows = glofas_events(thr) + [event_2026()]
    df = pd.concat([pd.DataFrame(rows), keep], ignore_index=True)
    save_labels(df, now)
    check = pd.read_csv(LABELS_CSV, dtype=str)
    assert list(check.columns) == COLUMNS and len(check) >= 2
    n_hi = int((check["confidence"] == "high").sum())
    print(f"OK data/labels/events.csv: {len(check)} events "
          f"({len(rows) - 1} GloFAS-q90 crossings, 1 reported event, {n_hi} CWC observed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
