"""FORECAST SCOREBOARD (2/2) - ready-made flood forecasts scored against
what actually happened, per station, per lead time, against persistence.

Sources scored (more can be added as they become available, e.g. the
Google Flood Forecasting API once pilot access is granted):
- cwc_aff : CWC Advisory Flood Forecast 7-day LEVEL forecasts (archived
            every 3 h with issue time) vs CWC WIMS OBSERVED hourly levels.
            Real observations -> this is the scoreboard that matters.
- glofas  : GloFAS 30-day ensemble median DISCHARGE forecasts (archived per
            run) vs GloFAS reanalysis. Modelled-vs-modelled: it measures
            GloFAS's own consistency, not skill against a river gauge, and
            says so.

Per source / station / lead bucket: n pairs, MAE, persistence MAE (value at
issue time carried forward), skill = 1 - MAE/MAE_persistence. Event framing
where a threshold exists (CWC official WARNING level; basin q90 for GloFAS):
hours/days above threshold observed vs forecast at the 24 h lead -> POD,
FAR. Everything is reported with n; thin archives say "insufficient".

Runs every pipeline pass; the numbers grow with the archive. Writes
data/history/scoreboard.json, public/forecast_scoreboard.json and
docs/FORECAST-SCOREBOARD.md (+ served copy).
"""

import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import DATA_DIR, OBSERVED, PUBLIC_DIR, REPO_ROOT, load_json, save_json, utc_now_iso  # noqa: E402

from basins import BASINS, model_names  # noqa: E402

HIST = DATA_DIR / "history"
DOCS = REPO_ROOT / "docs"
CWC_LEADS_H = {"6h": (3, 9), "12h": (9, 18), "24h": (18, 36), "48h": (36, 60),
               "72h": (60, 96), "7d": (96, 192)}
GLOFAS_LEADS_D = [1, 2, 3, 5, 7]
MIN_PAIRS = 12


def pv(value, source, now, **extra):
    return {"value": value, "class": OBSERVED, "source": source, "retrieved_at": now, **extra}


def score(pairs: pd.DataFrame) -> dict:
    """pairs: columns fc, obs, pers. Returns metrics or insufficient marker."""
    pairs = pairs.dropna(subset=["fc", "obs", "pers"])
    n = int(len(pairs))
    if n < MIN_PAIRS:
        return {"n": n, "status": "insufficient"}
    mae = float(np.mean(np.abs(pairs["fc"] - pairs["obs"])))
    mp = float(np.mean(np.abs(pairs["pers"] - pairs["obs"])))
    return {"n": n, "status": "ok", "mae": round(mae, 3), "mae_persistence": round(mp, 3),
            "skill_vs_persistence": round(1 - mae / mp, 3) if mp > 0 else None,
            "bias": round(float(np.mean(pairs["fc"] - pairs["obs"])), 3)}


def events(pairs: pd.DataFrame, thr: float) -> dict:
    p = pairs.dropna(subset=["fc", "obs"])
    if thr is None or p.empty:
        return {"status": "no threshold" if thr is None else "no pairs"}
    obs_ab, fc_ab = p["obs"] >= thr, p["fc"] >= thr
    n_obs, n_fc = int(obs_ab.sum()), int(fc_ab.sum())
    if n_obs == 0 and n_fc == 0:
        return {"status": "no crossings observed or forecast yet", "n": int(len(p)), "threshold": thr}
    return {"status": "ok", "n": int(len(p)), "threshold": thr,
            "observed_above": n_obs, "forecast_above": n_fc,
            "pod": round(float((obs_ab & fc_ab).sum() / n_obs), 3) if n_obs else None,
            "far": round(float((fc_ab & ~obs_ab).sum() / n_fc), 3) if n_fc else None}


def score_cwc(now: str) -> dict:
    out = {}
    d = HIST / "cwc_aff"
    marks = load_json(d / "stations.json")["stations"] if (d / "stations.json").exists() else {}
    for fcp in sorted((d / "forecasts").glob("*.parquet")):
        st = fcp.stem
        obp = d / "observed" / fcp.name
        if not obp.exists():
            continue
        fc = pd.read_parquet(fcp).dropna(subset=["time_utc", "issued_at_utc"])
        ob = pd.read_parquet(obp).dropna(subset=["time_utc"])
        fc["t"] = pd.to_datetime(fc["time_utc"], utc=True)
        fc["i"] = pd.to_datetime(fc["issued_at_utc"], utc=True)
        ob["t"] = pd.to_datetime(ob["time_utc"], utc=True)
        obs = ob.set_index("t")["wims_level_m"].sort_index()
        fc["obs"] = fc["t"].map(obs)
        # persistence: last observation at or before issue time
        fc = fc.sort_values("i")
        pers = pd.merge_asof(fc[["i"]].rename(columns={"i": "t"}).sort_values("t"),
                             obs.reset_index().rename(columns={"wims_level_m": "pers"}).sort_values("t"),
                             on="t", direction="backward")["pers"].to_numpy()
        fc["pers"] = pers
        fc["lead_h"] = (fc["t"] - fc["i"]).dt.total_seconds() / 3600
        fc = fc.rename(columns={"aff_level_m": "fc"})
        key = next((k for k in marks if re.sub(r"[^A-Za-z0-9]+", "_", k).strip("_") == st), None)
        warn = (marks.get(key) or {}).get("warning_m") if key else None
        leads = {}
        for lb, (lo, hi) in CWC_LEADS_H.items():
            sel = fc[(fc["lead_h"] >= lo) & (fc["lead_h"] < hi)][["fc", "obs", "pers"]]
            leads[lb] = score(sel)
        ev = events(fc[(fc["lead_h"] >= 18) & (fc["lead_h"] < 36)][["fc", "obs"]], warn)
        out[st] = {"station": key or st, "warning_level_m": warn,
                   "issue_times": int(fc["i"].nunique()), "observed_hours": int(len(obs)),
                   "leads": leads, "warning_events_24h": ev}
    return out


def score_glofas(now: str) -> dict:
    out = {}
    d = HIST / "glofas"
    if not d.exists():
        return out
    thr_by_cell = {}
    for b in BASINS:
        mp = REPO_ROOT / "models" / model_names(b)["meta"]
        if mp.exists():
            thr_by_cell[BASINS[b]["target"]] = load_json(mp)["threshold_q90_monsoon_2015_2025_m3s"]
    for cell in sorted(p.name for p in d.iterdir() if p.is_dir()):
        fcp, obp = d / cell / "forecasts.parquet", d / cell / "observed.parquet"
        if not (fcp.exists() and obp.exists()):
            continue
        fc = pd.read_parquet(fcp)
        ob = pd.read_parquet(obp)
        obs = ob.set_index(pd.to_datetime(ob["date"]))["discharge_m3s"].sort_index()
        fc["t"] = pd.to_datetime(fc["target_date"]); fc["i"] = pd.to_datetime(fc["issued_date"])
        fc["obs"] = fc["t"].map(obs)
        fc["pers"] = fc["i"].map(obs)
        fc["lead_d"] = (fc["t"] - fc["i"]).dt.days
        fc = fc.rename(columns={"median_m3s": "fc"})
        leads = {f"{L}d": score(fc[fc["lead_d"] == L][["fc", "obs", "pers"]]) for L in GLOFAS_LEADS_D}
        ev = events(fc[fc["lead_d"] == 1][["fc", "obs"]], thr_by_cell.get(cell))
        out[cell] = {"issue_dates": int(fc["i"].nunique()), "observed_days": int(len(obs)),
                     "leads": leads, "q90_events_1d": ev}
    return out


def main() -> int:
    now = utc_now_iso()
    cwc, glo = score_cwc(now), score_glofas(now)
    src_c = "derived: CWC AFF forecast level vs CWC WIMS observed level (real gauge), per lead bucket"
    src_g = "derived: GloFAS ensemble median forecast vs GloFAS reanalysis (modelled vs modelled)"

    def wrap(block, src):
        w = {}
        for k, v in block.items():
            w[k] = {"leads": {lb: pv(m.get("mae"), src, now, **{kk: vv for kk, vv in m.items() if kk != "mae"})
                              for lb, m in v["leads"].items()},
                    **{kk: vv for kk, vv in v.items() if kk != "leads"}}
        return w

    doc = {"generated_at": now,
           "note": ("Ready-made flood forecasts scored against what happened. cwc_aff is scored "
                    "against REAL observed gauge levels; glofas is scored against GloFAS's own "
                    "reanalysis (modelled vs modelled). Numbers accumulate with the archive; "
                    f"buckets with fewer than {MIN_PAIRS} pairs are marked insufficient. "
                    "The Google Flood Forecasting API will be added as a source when pilot "
                    "access is granted."),
           "min_pairs": MIN_PAIRS,
           "sources": {"cwc_aff": wrap(cwc, src_c), "glofas": wrap(glo, src_g)}}
    save_json(HIST / "scoreboard.json", doc)
    save_json(PUBLIC_DIR / "forecast_scoreboard.json", doc)

    L = []
    A = L.append
    A("# FORECAST-SCOREBOARD - ready-made forecasts vs what happened")
    A("")
    A(f"Generated {now} by `backend/model/scoreboard.py`; regenerated every pipeline run.")
    A("Skill = 1 − MAE/MAE(persistence); persistence = the value at issue time carried")
    A(f"forward. Buckets with < {MIN_PAIRS} pairs are 'insufficient'. Grows with the archive.")
    A("")
    A("## CWC Advisory Flood Forecast (levels, m) vs CWC observed gauge levels - REAL observations")
    A("")
    A("| station | issues | obs hours | " + " | ".join(f"{lb} n / MAE / skill" for lb in CWC_LEADS_H) + " | warning events @24h |")
    A("|---|---|---|" + "---|" * len(CWC_LEADS_H) + "---|")
    for st, v in cwc.items():
        cells = []
        for lb in CWC_LEADS_H:
            m = v["leads"][lb]
            cells.append(f"{m['n']} / {m['mae']} / {m['skill_vs_persistence']}" if m["status"] == "ok" else f"{m['n']} / – / –")
        ev = v["warning_events_24h"]
        evs = (f"POD {ev['pod']} FAR {ev['far']} ({ev['observed_above']} h above {ev['threshold']} m)"
               if ev.get("status") == "ok" else ev.get("status"))
        A(f"| {v['station']} | {v['issue_times']} | {v['observed_hours']} | " + " | ".join(cells) + f" | {evs} |")
    A("")
    A("## GloFAS ensemble median (discharge, m³/s) vs GloFAS reanalysis - modelled vs modelled")
    A("")
    A("| cell | issues | obs days | " + " | ".join(f"{L}d n / MAE / skill" for L in GLOFAS_LEADS_D) + " | q90 events @1d |")
    A("|---|---|---|" + "---|" * len(GLOFAS_LEADS_D) + "---|")
    for cell, v in glo.items():
        cells = []
        for Ld in GLOFAS_LEADS_D:
            m = v["leads"][f"{Ld}d"]
            cells.append(f"{m['n']} / {m['mae']} / {m['skill_vs_persistence']}" if m["status"] == "ok" else f"{m['n']} / – / –")
        ev = v["q90_events_1d"]
        evs = (f"POD {ev['pod']} FAR {ev['far']}" if ev.get("status") == "ok" else ev.get("status"))
        A(f"| {cell} | {v['issue_dates']} | {v['observed_days']} | " + " | ".join(cells) + f" | {evs} |")
    A("")
    A("## Reading this honestly")
    A("")
    A("- CWC levels are the only REAL observations here; GloFAS is scored against its own")
    A("  reanalysis, which measures consistency, not truth.")
    A("- A forecast that cannot beat persistence at a lead has no value at that lead -")
    A("  that is the bar, exactly as for the project's own models.")
    A("- Event stats need crossings to have happened; early in the archive they will")
    A("  read 'no crossings observed or forecast yet'. That is not a result, it is a wait.")
    A("- Sources to add when access arrives: Google Flood Forecasting API (pilot).")
    A("")
    md = "\n".join(L) + "\n"
    DOCS.mkdir(exist_ok=True)
    (DOCS / "FORECAST-SCOREBOARD.md").write_text(md, encoding="utf-8")
    (PUBLIC_DIR / "FORECAST-SCOREBOARD.md").write_text(md, encoding="utf-8")
    n_ok = sum(1 for v in cwc.values() for m in v["leads"].values() if m["status"] == "ok")
    print(f"OK forecast scoreboard: {len(cwc)} CWC stations, {len(glo)} GloFAS cells; "
          f"{n_ok} CWC lead buckets with >= {MIN_PAIRS} pairs so far")
    return 0


if __name__ == "__main__":
    sys.exit(main())
