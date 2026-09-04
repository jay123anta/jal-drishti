"""TRACK T2 - CWC official gauges from the Advisory Flood Forecast public
dissemination portal (aff.india-water.gov.in).

Scope rule: the portal serves plain
public data files (with an official CSV download) as a flood-dissemination
product. Conditions honoured here:
- attribution to CWC on every value;
- fetch cadence capped at the portal's own 3-hourly update cycle
  (FETCH-STATE.json freshness guard; --force overrides);
- local archive, because the portal keeps only a rolling ~2-day observed
  window (each fetch appends new hours; history accumulates from today);
- honest degradation on any format change or outage (stale values are
  marked stale; nothing is fabricated).

Files read (public, no auth):
  /textdata/Floodday_table_view_header.txt  -> official WARNING / DANGER /
      HFL marks, coordinates, latest observed level + condition
  /Timeseries/<STATION>-W.txt               -> hourly: WIMS observed level
      (class OBSERVED), AFF model forecast level (class FORECAST), CWC-
      derived discharge (reference), catchment rainfall

Archive:
  data/history/cwc_aff/observed/<STATION>.parquet  (+ sidecar, OBSERVED)
  data/history/cwc_aff/forecasts/<STATION>.parquet (+ sidecar, FORECAST,
      archived:true; each row keeps its issue time -> CWC's own forecast
      skill can be evaluated later)
  data/history/cwc_aff/stations.json, FETCH-STATE.json
  data/history/cwc_aff/raw/<STATION>-W.latest.txt (latest response only;
      not accumulated, to keep the repo small - the parquet is the record)

Payload: public/cwc_stations.json - per station, provenance-wrapped:
official marks, latest observed level, margin to warning, CWC 7-day peak.

Times: the portal publishes IST; both IST strings and UTC are stored.
"""

import csv
import datetime
import io
import pathlib
import re
import sys
import urllib.parse

import pandas as pd
import requests

from common import (DATA_DIR, FORECAST, OBSERVED, PUBLIC_DIR, load_json,
                    save_json, utc_now_iso)

AFF = "https://aff.india-water.gov.in"
UA = "JalDrishti-PoC/1.0 (research feasibility demo; fetch capped at 3-hourly)"
MIN_INTERVAL_H = 3
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

SRC = ("Central Water Commission (CWC), Ministry of Jal Shakti - Advisory "
       "Flood Forecast public dissemination portal (aff.india-water.gov.in)")
SRC_MARKS = SRC + " - official warning/danger/HFL marks (station metadata, unclassified per National Water Data Policy 4.2.3)"
SRC_OBS = SRC + " - WIMS hourly observed water level (gauge telemetry)"
SRC_FC = SRC + " - AFF 7-day model forecast level (CWC's own forecast, updated 3-hourly)"
SRC_Q = SRC + " - CWC-derived discharge (reference only; rating curves are classified)"

# CWC AFF station -> this PoC's river point + a plain-language name
STATIONS = [
    {"aff": "SIVASAGAR", "poc_river": "dikhow_sivasagar", "plain": "Sivasagar (Dikhow)"},
    {"aff": "NANGLAMORAGHAT", "poc_river": "disang_nanglamoraghat", "plain": "Nanglamoraghat (Disang)"},
    {"aff": "KAMPUR", "poc_river": "kopili_kampur", "plain": "Kampur (Kopili)"},
    {"aff": "DHARAMTUL", "poc_river": None, "plain": "Dharamtul (Kopili)"},
    {"aff": "NT ROAD CROSSING JIA-BHARALI", "poc_river": "jiabharali_tezpur", "plain": "NT Road crossing (Jia Bharali)"},
    {"aff": "DIBRUGARH", "poc_river": "brahmaputra_dibrugarh", "plain": "Dibrugarh (Brahmaputra)"},
    {"aff": "TEZPUR", "poc_river": "brahmaputra_tezpur", "plain": "Tezpur (Brahmaputra)"},
    {"aff": "GUWAHATI(D.C.COURT)", "poc_river": "brahmaputra_guwahati", "plain": "Guwahati (Brahmaputra)"},
    {"aff": "BEKI ROAD BRIDGE", "poc_river": "beki_barpeta", "plain": "Beki road bridge (Beki)"},
    {"aff": "ANNAPURNA GHAT", "poc_river": "barak_silchar", "plain": "Annapurna Ghat (Barak, Silchar)"},
    {"aff": "GOLAGHAT", "poc_river": "dhansiri_golaghat", "plain": "Golaghat (Dhansiri)"},
    {"aff": "BADATIGHAT", "poc_river": "subansiri_badatighat", "plain": "Badatighat (Subansiri)"},
    {"aff": "CHOULDHOWAGHAT", "poc_river": None, "plain": "Chouldhowaghat (Subansiri, upstream)"},
    {"aff": "MANAS N H CROSSING", "poc_river": "manas_nhcrossing", "plain": "Manas NH crossing"},
    {"aff": "RANGANADI NT ROAD CROSSING", "poc_river": "ranganadi_ntxing", "plain": "NT Road crossing (Ranganadi)"},
    {"aff": "MATIJURI", "poc_river": "katakhal_matijuri", "plain": "Matijuri (Katakhal)"},
    {"aff": "GOLOKGANJ", "poc_river": "sankosh_golokganj", "plain": "Golokganj (Sankosh)"},
    {"aff": "NUMALIGARH", "poc_river": None, "plain": "Numaligarh (Dhansiri)"},
]

HIST = DATA_DIR / "history" / "cwc_aff"
OBS_DIR, FC_DIR, RAW_DIR = HIST / "observed", HIST / "forecasts", HIST / "raw"
STATE_PATH = HIST / "FETCH-STATE.json"
STATIONS_PATH = HIST / "stations.json"
PAYLOAD_PATH = PUBLIC_DIR / "cwc_stations.json"


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def pv(value, cls, source, retrieved_at, **extra):
    return {"value": value, "class": cls, "source": source,
            "retrieved_at": retrieved_at, **extra}


def ist_to_utc(s: str) -> str | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%y %H:%M"):
        try:
            t = datetime.datetime.strptime(s.strip(), fmt).replace(tzinfo=IST)
            return t.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        except ValueError:
            continue
    return None


def get(path: str, timeout: int = 60) -> str:
    last = None
    for attempt in range(3):
        try:
            r = requests.get(AFF + path, headers={"User-Agent": UA}, timeout=timeout)
            r.raise_for_status()
            return r.text
        except requests.RequestException as err:
            last = err
            import time
            time.sleep(3 * (2 ** attempt))
    raise RuntimeError(f"GET {path} failed after 3 attempts: {last}")


def fnum(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def parse_table(text: str) -> dict:
    """Station marks + latest observed level from the flood table file.
    Only the header-named leading columns are used (rows carry extra
    trailing forecast fields, taken from the timeseries file instead)."""
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}
    out = {}
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        name = r[idx["Station"]].strip()
        out[name] = {
            "river": r[idx["River"]], "district": r[idx["District"]],
            "state": r[idx["State"]],
            "lat": fnum(r[idx["Latitude"]]), "lon": fnum(r[idx["Longitude"]]),
            "warning_m": fnum(r[idx["WarningLevel"]]),
            "danger_m": fnum(r[idx["DangerLevel"]]), "hfl_m": fnum(r[idx["HFL"]]),
            "observed_at_ist": r[idx["Date_WIMS"]], "observed_m": fnum(r[idx["WIMS_Value"]]),
            "condition": r[idx["current_condition"]],
            "table_time_ist": r[idx["timeofforecast"]],
        }
    return out


def merge_parquet(path: pathlib.Path, new: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = df.drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def archive_health(obs_all: pd.DataFrame) -> dict:
    """Hourly continuity of the observed archive: missing hours between the
    first and last stamp, and the longest gap. Surfaced in the payload and
    the Technical view so a broken keep-alive is visible, not silent."""
    ts = pd.to_datetime(obs_all["time_utc"].dropna().unique(), utc=True)
    if len(ts) < 2:
        return {"missing_hours": 0, "longest_gap_h": 0, "span_hours": len(ts)}
    ts = ts.sort_values()
    span_h = int((ts[-1] - ts[0]).total_seconds() // 3600) + 1
    missing = int(span_h - len(ts))
    diffs = (ts[1:] - ts[:-1]).total_seconds() / 3600
    longest = int(max(diffs)) - 1 if len(diffs) else 0
    return {"missing_hours": max(missing, 0), "longest_gap_h": max(longest, 0),
            "span_hours": span_h}


def sidecar(path: pathlib.Path, df: pd.DataFrame, cls: str, source: str,
            retrieved_at: str, tcol: str, extra: dict) -> None:
    save_json(path.with_name(path.stem + ".provenance.json"), {
        "station_file": path.name, "rows": int(len(df)),
        "first": str(df[tcol].min()) if len(df) else None,
        "last": str(df[tcol].max()) if len(df) else None,
        "columns": list(df.columns), "class": cls, "source": source,
        "retrieved_at": retrieved_at,
        "note": ("times: *_ist as published (Asia/Kolkata), *_utc derived; "
                 "archive accumulates from the first fetch - the portal keeps "
                 "only a rolling ~2-day observed window"),
        **extra,
    })


def process_station(st: dict, table: dict, retrieved_at: str) -> dict:
    name = st["aff"]
    text = get("/Timeseries/" + urllib.parse.quote(name) + "-W.txt")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{safe(name)}-W.latest.txt").write_text(text, encoding="utf-8")
    df = pd.read_csv(io.StringIO(text))
    need = {"Date", "AFF", "WIMS", "TOF"}
    if not need <= set(df.columns):
        raise RuntimeError(f"format change: columns {list(df.columns)[:8]}...")
    tof = str(df["TOF"].iloc[0])

    obs = df[df["WIMS"].notna()].copy()
    obs_out = pd.DataFrame({
        "time_ist": obs["Date"].astype(str),
        "time_utc": obs["Date"].astype(str).map(ist_to_utc),
        "wims_level_m": obs["WIMS"].astype(float),
        "cwc_discharge_m3s": obs.get("Discharge"),
        "catchment_rain_mm": obs.get("RainfallAverageCatchement"),
    })
    obs_all = merge_parquet(OBS_DIR / f"{safe(name)}.parquet", obs_out, ["time_utc"])
    sidecar(OBS_DIR / f"{safe(name)}.parquet", obs_all, OBSERVED, SRC_OBS, retrieved_at,
            "time_utc", {"discharge_note": SRC_Q, "station": name})

    fc = df[df["Date"].astype(str) > tof].copy()
    fc_out = pd.DataFrame({
        "issued_at_ist": tof, "issued_at_utc": ist_to_utc(tof),
        "time_ist": fc["Date"].astype(str),
        "time_utc": fc["Date"].astype(str).map(ist_to_utc),
        "aff_level_m": fc["AFF"].astype(float),
        "cwc_discharge_m3s": fc.get("Discharge"),
    })
    fc_all = merge_parquet(FC_DIR / f"{safe(name)}.parquet", fc_out,
                           ["issued_at_utc", "time_utc"])
    sidecar(FC_DIR / f"{safe(name)}.parquet", fc_all, FORECAST, SRC_FC, retrieved_at,
            "time_utc", {"archived": True, "station": name})

    meta = table.get(name, {})
    latest = obs_out.sort_values("time_utc").iloc[-1] if len(obs_out) else None
    warn = meta.get("warning_m")
    peak_row = fc_out.loc[fc_out["aff_level_m"].idxmax()] if len(fc_out) else None
    cross = None
    if warn is not None and len(fc_out):
        above = fc_out[fc_out["aff_level_m"] >= warn]
        if len(above):
            cross = str(above.iloc[0]["time_ist"])
    # official level trends from OUR archive of the gauge's own readings:
    # a short "right now" window (~6 h, what ASDMA alerts react to) and a
    # 24 h context window.
    def _trend(hours, near, far, thresh):
        if latest is None or len(obs_all) < 2:
            return None, None, None
        oa = obs_all.sort_values("time_utc")
        t_latest = pd.Timestamp(str(latest["time_utc"]))
        target = t_latest - pd.Timedelta(hours=hours)
        older = oa[pd.to_datetime(oa["time_utc"]) <= t_latest - pd.Timedelta(hours=near)]
        if not len(older):
            return None, None, None
        ref = older.iloc[(pd.to_datetime(older["time_utc"]) - target).abs().argmin()]
        if abs((pd.Timestamp(str(ref["time_utc"])) - target).total_seconds()) > far * 3600:
            return None, None, None
        delta = round(float(latest["wims_level_m"]) - float(ref["wims_level_m"]), 2)
        word = "rising" if delta >= thresh else "falling" if delta <= -thresh else "steady"
        return word, delta, str(ref["time_utc"])
    trend_word, trend_delta, trend_ref_utc = _trend(24, 18, 8, 0.05)
    now_word, now_delta, now_ref_utc = _trend(6, 3, 5, 0.03)
    return {
        "meta": meta, "tof": tof,
        "obs_trend_word": trend_word, "obs_trend_delta_m": trend_delta,
        "obs_trend_ref_utc": trend_ref_utc,
        "obs_now_word": now_word, "obs_now_delta_m": now_delta,
        "obs_now_ref_utc": now_ref_utc,
        "latest_level": float(latest["wims_level_m"]) if latest is not None else None,
        "latest_at_ist": str(latest["time_ist"]) if latest is not None else None,
        "latest_at_utc": str(latest["time_utc"]) if latest is not None else None,
        "latest_q": (float(latest["cwc_discharge_m3s"]) if latest is not None
                     and pd.notna(latest["cwc_discharge_m3s"]) else None),
        "peak_m": float(peak_row["aff_level_m"]) if peak_row is not None else None,
        "peak_at_ist": str(peak_row["time_ist"]) if peak_row is not None else None,
        "crosses_warning_at_ist": cross,
        "n_obs_archive": int(len(obs_all)), "archive_first_utc": str(obs_all["time_utc"].min()),
        "archive_last_utc": str(obs_all["time_utc"].max()),
        "n_fc_archive": int(len(fc_all)),
        "health": archive_health(obs_all),
    }


def station_entry(st: dict, res: dict | None, retrieved_at: str, error: str | None,
                  prev: dict | None) -> dict:
    """Payload entry; falls back to the previous payload entry marked stale."""
    if res is None:
        base = dict(prev or {"aff_station": st["aff"], "poc_river": st["poc_river"],
                             "plain_name": st["plain"]})
        base.update({"degraded": True, "stale": prev is not None, "error": error})
        return base
    m = res["meta"]
    e = {
        "aff_station": st["aff"], "poc_river": st["poc_river"], "plain_name": st["plain"],
        "river": m.get("river"), "district": m.get("district"),
        "lat": m.get("lat"), "lon": m.get("lon"),
        "warning_level_m": pv(m.get("warning_m"), OBSERVED, SRC_MARKS, retrieved_at, unit="m"),
        "danger_level_m": pv(m.get("danger_m"), OBSERVED, SRC_MARKS, retrieved_at, unit="m"),
        "hfl_m": pv(m.get("hfl_m"), OBSERVED, SRC_MARKS, retrieved_at, unit="m",
                    note="highest flood level on record"),
        "observed_level_m": pv(res["latest_level"], OBSERVED, SRC_OBS, retrieved_at, unit="m",
                               observed_at_ist=res["latest_at_ist"],
                               observed_at_utc=res["latest_at_utc"]),
        "cwc_discharge_m3s": pv(res["latest_q"], OBSERVED, SRC_Q, retrieved_at, unit="m3/s",
                                note="reference only, CWC-derived"),
        "condition": m.get("condition"),
        "margin_to_warning_m": pv(
            (round(m["warning_m"] - res["latest_level"], 2)
             if m.get("warning_m") is not None and res["latest_level"] is not None else None),
            OBSERVED, "derived: official warning level minus latest observed level",
            retrieved_at, unit="m"),
        "observed_trend_now": pv(res["obs_now_word"], OBSERVED,
                                 "derived from this project's archive of CWC observed levels "
                                 "(latest reading vs the reading ~6 h earlier)",
                                 retrieved_at, delta_m=res["obs_now_delta_m"],
                                 compared_with_utc=res["obs_now_ref_utc"]),
        "observed_trend_24h": pv(res["obs_trend_word"], OBSERVED,
                                 "derived from this project's archive of CWC observed levels "
                                 "(latest reading vs the reading ~24 h earlier)",
                                 retrieved_at, delta_m=res["obs_trend_delta_m"],
                                 compared_with_utc=res["obs_trend_ref_utc"]),
        "cwc_forecast_peak_m": pv(res["peak_m"], FORECAST, SRC_FC, retrieved_at, unit="m",
                                  issued_at_ist=res["tof"], peak_at_ist=res["peak_at_ist"]),
        "cwc_forecast_crosses_warning_at_ist": res["crosses_warning_at_ist"],
        "archive": {"observed_rows": res["n_obs_archive"],
                    "observed_first_utc": res["archive_first_utc"],
                    "observed_last_utc": res["archive_last_utc"],
                    "forecast_rows": res["n_fc_archive"],
                    **res["health"]},
        "degraded": False, "stale": False, "error": None,
    }
    return e


def main() -> int:
    now = utc_now_iso()
    state = load_json(STATE_PATH) if STATE_PATH.exists() else {}
    prev_payload = load_json(PAYLOAD_PATH) if PAYLOAD_PATH.exists() else {}
    prev_by = {s["aff_station"]: s for s in prev_payload.get("stations", [])}

    last = state.get("last_fetch_at")
    if last and "--force" not in sys.argv:
        age_h = (datetime.datetime.now(datetime.timezone.utc)
                 - datetime.datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ")
                 .replace(tzinfo=datetime.timezone.utc)).total_seconds() / 3600
        if age_h < MIN_INTERVAL_H and PAYLOAD_PATH.exists():
            print(f"OK CWC AFF: last fetch {age_h:.1f} h ago (< {MIN_INTERVAL_H} h cap) - "
                  f"not re-fetched; payload kept (--force to override)")
            return 0

    try:
        table = parse_table(get("/textdata/Floodday_table_view_header.txt"))
    except (RuntimeError, KeyError, IndexError) as err:
        table = {}
        print(f"  station table FAILED: {err}")

    entries, degraded = [], []
    for st in STATIONS:
        try:
            res = process_station(st, table, now)
            entries.append(station_entry(st, res, now, None, None))
            print(f"  {st['aff']:30s} obs {res['latest_level']} m @ {res['latest_at_ist']} "
                  f"({(table.get(st['aff']) or {}).get('condition')}), warn "
                  f"{(table.get(st['aff']) or {}).get('warning_m')}, CWC 7d peak "
                  f"{res['peak_m']} | archive {res['n_obs_archive']} obs rows")
        except (RuntimeError, ValueError, KeyError, IndexError) as err:
            degraded.append(st["aff"])
            entries.append(station_entry(st, None, now, str(err), prev_by.get(st["aff"])))
            print(f"  {st['aff']:30s} DEGRADED: {err}")

    # label store hook: observed WARNING-level crossings become high-confidence events
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "model"))
        import build_labels
        n_new = build_labels.cwc_crossings_to_labels(STATIONS, OBS_DIR, table, now)
        if n_new:
            print(f"  labels: +{n_new} observed warning-level crossing(s) -> data/labels/events.csv")
    except Exception as err:  # noqa: BLE001 - labels must never break the fetch
        print(f"  labels hook skipped: {err}")

    save_json(STATIONS_PATH, {"retrieved_at": now, "source": SRC_MARKS,
                              "stations": {k: v for k, v in table.items()
                                           if k in {s['aff'] for s in STATIONS}}})
    save_json(PAYLOAD_PATH, {
        "generated_at": now,
        "source": SRC,
        "attribution": ("Data: Central Water Commission, Ministry of Jal Shakti, "
                        "Government of India (public flood dissemination). "
                        "Levels are official gauge readings; forecasts are CWC's."),
        "fetch_policy": (f"fetched at most every {MIN_INTERVAL_H} h (portal update "
                         f"cadence); documented project rules"),
        "last_fetch_at": now,
        "degraded_stations": degraded,
        "archive_health": {
            "stations_with_gaps": [e["aff_station"] for e in entries
                                   if not e.get("degraded") and e["archive"]["missing_hours"] > 0],
            "worst_missing_hours": max([e["archive"]["missing_hours"] for e in entries
                                        if not e.get("degraded")] or [0]),
            "note": ("missing hours between first and last archived stamp per "
                     "station; a keep-alive gap > 48 h loses data for good "
                     "(the portal keeps ~2 days)"),
        },
        "stations": entries,
    })
    if len(degraded) < len(STATIONS):
        state["last_fetch_at"] = now
    state.setdefault("fetch_count", 0)
    state["fetch_count"] += 1
    state["last_degraded"] = degraded
    save_json(STATE_PATH, state)

    ok = len(STATIONS) - len(degraded)
    print(f"OK public/cwc_stations.json: {ok}/{len(STATIONS)} stations live"
          f"{', degraded: ' + ', '.join(degraded) if degraded else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
