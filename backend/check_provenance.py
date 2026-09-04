"""Provenance gate: fails (exit 1) if any exported value is missing
source / retrieved_at / class, or if risk values are not labelled SIMULATED.

Checked files: public/villages_status.json, public/rivers_status.json,
data/rainfall.json, data/discharge.json, and (Upgrade v2)
data/archive/2026-07/*.json, public/replay_2026-07.json,
public/replay_findings.json, public/flood_extent_*.geojson (if present).

Rules enforced:
- any dict with a "value" key is a measurement -> needs source, retrieved_at,
  class in {OBSERVED, FORECAST, SIMULATED}
- any record inside an "hourly"/"daily" list -> same requirement
- every village risk: value in {GREEN,YELLOW,RED}, class SIMULATED, label
  contains "UNVALIDATED"
- top-level disclaimer present on both public payloads
- every replay risk: class SIMULATED, replay true, label UNVALIDATED; replay
  inputs class OBSERVED; archive series all class OBSERVED
- flood-extent GeoJSON features: class OBSERVED with scene id, acquisition
  date, method and threshold in properties
"""

import sys

from common import DATA_DIR, PUBLIC_DIR, load_json

VALID_CLASSES = {"OBSERVED", "FORECAST", "SIMULATED"}
errors: list[str] = []


def need_provenance(obj: dict, path: str):
    for field in ("source", "retrieved_at", "class"):
        if field not in obj:
            errors.append(f"{path}: missing {field}")
    if obj.get("class") not in VALID_CLASSES:
        errors.append(f"{path}: bad class {obj.get('class')!r}")


def walk(node, path: str):
    if isinstance(node, dict):
        if "value" in node:
            need_provenance(node, path)
        for k, val in node.items():
            if k in ("hourly", "daily") and isinstance(val, list):
                for i, rec in enumerate(val):
                    if isinstance(rec, dict):
                        need_provenance(rec, f"{path}.{k}[{i}]")
            else:
                walk(val, f"{path}.{k}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            walk(item, f"{path}[{i}]")


def check_replay(rep: dict):
    if rep.get("replay") is not True:
        errors.append("replay: top-level replay flag missing/false")
    if "UNVALIDATED" not in rep.get("disclaimer", ""):
        errors.append("replay: top-level disclaimer missing")
    for v in rep.get("villages", []):
        for i, e in enumerate(v.get("steps", [])):
            where = f"replay.{v['name']}[{i}]"
            r = e.get("risk", {})
            if r.get("value") not in {"GREEN", "YELLOW", "RED"}:
                errors.append(f"{where}: bad risk value {r.get('value')!r}")
            if r.get("class") != "SIMULATED":
                errors.append(f"{where}: risk class must be SIMULATED")
            if r.get("replay") is not True:
                errors.append(f"{where}: risk missing replay:true")
            if "UNVALIDATED" not in r.get("label", ""):
                errors.append(f"{where}: risk label missing UNVALIDATED")
            for k in ("rain24", "rain48", "discharge_m3s", "discharge_pctl"):
                if e.get(k, {}).get("class") != "OBSERVED":
                    errors.append(f"{where}: input {k} must be class OBSERVED")


def check_archive(doc: dict, name: str, series: str):
    for pt in doc.get("points", []):
        for i, rec in enumerate(pt.get(series, [])):
            if rec.get("class") != "OBSERVED":
                errors.append(f"{name}.{pt['id']}[{i}]: archive class must be OBSERVED")


def check_extent(doc: dict, name: str):
    feats = doc.get("features", [])
    if not feats:
        errors.append(f"{name}: no features")
    for i, f in enumerate(feats):
        p = f.get("properties", {})
        if p.get("class") != "OBSERVED":
            errors.append(f"{name}[{i}]: extent class must be OBSERVED")
        for field in ("scene_id", "acquired", "method", "source", "retrieved_at"):
            if not p.get(field):
                errors.append(f"{name}[{i}]: missing {field}")


def main() -> int:
    vs = load_json(PUBLIC_DIR / "villages_status.json")
    rs = load_json(PUBLIC_DIR / "rivers_status.json")
    rain = load_json(DATA_DIR / "rainfall.json")
    disch = load_json(DATA_DIR / "discharge.json")

    for name, doc in [("villages_status", vs), ("rivers_status", rs),
                      ("rainfall", rain), ("discharge", disch)]:
        walk(doc, name)

    for doc, name in ((vs, "villages_status"), (rs, "rivers_status")):
        if "UNVALIDATED" not in doc.get("disclaimer", ""):
            errors.append(f"{name}: top-level disclaimer missing")

    for v in vs["villages"]:
        r = v["risk"]
        if r["value"] not in {"GREEN", "YELLOW", "RED"}:
            errors.append(f"{v['name']}: bad risk value {r['value']!r}")
        if r["class"] != "SIMULATED":
            errors.append(f"{v['name']}: risk class must be SIMULATED, got {r['class']!r}")
        if "UNVALIDATED" not in r.get("label", ""):
            errors.append(f"{v['name']}: risk label missing UNVALIDATED marker")
        if not (r.get("method") == "heuristic" or str(r.get("method", "")).startswith("model-v0-")):
            errors.append(f"{v['name']}: risk method missing/bad: {r.get('method')!r}")
        if str(r.get("method", "")).startswith("model-v0-"):
            for field in ("forecast_horizon", "basis", "target_caveat"):
                if not r.get(field):
                    errors.append(f"{v['name']}: model risk missing {field}")
        for key in ("rain_trailing_24h_mm", "rain_trailing_48h_mm",
                    "discharge_latest_m3s", "discharge_percentile_30d"):
            if key not in v["signals"]:
                errors.append(f"{v['name']}: signal {key} missing")

    # ---- Upgrade v2 payloads ----
    n_extra = []
    arch_dir = DATA_DIR / "archive" / "2026-07"
    for fname, series in (("archive_rainfall.json", "hourly"),
                          ("archive_discharge.json", "daily")):
        path = arch_dir / fname
        if path.exists():
            doc = load_json(path)
            walk(doc, fname)
            check_archive(doc, fname, series)
            n_extra.append(fname)
        else:
            errors.append(f"{fname}: missing (run fetch_archive.py)")

    rep_path = PUBLIC_DIR / "replay_2026-07.json"
    if rep_path.exists():
        rep = load_json(rep_path)
        walk(rep, "replay")
        check_replay(rep)
        n_extra.append("replay_2026-07.json")
    else:
        errors.append("replay_2026-07.json: missing (run classify_risk.py --replay)")

    rf_path = PUBLIC_DIR / "replay_findings.json"
    if rf_path.exists():
        rf = load_json(rf_path)
        walk(rf, "replay_findings")
        if rf.get("replay") is not True or "UNVALIDATED" not in rf.get("disclaimer", ""):
            errors.append("replay_findings: replay flag / disclaimer missing")
        n_extra.append("replay_findings.json")
    else:
        errors.append("replay_findings.json: missing (run analyze_replay.py)")

    # flood extents are OPTIONAL (Step B may be degraded) but validated if present
    for fname in ("flood_extent_2022.geojson", "flood_extent_2026.geojson"):
        path = PUBLIC_DIR / fname
        if path.exists():
            check_extent(load_json(path), fname)
            n_extra.append(fname)

    # Model v0 payloads, generic per basin: live forecast + 2026 out-of-sample test
    import importlib.util as _ilu2
    _bspec = _ilu2.spec_from_file_location("basins", DATA_DIR.parent / "backend" / "model" / "basins.py")
    _basins = _ilu2.module_from_spec(_bspec); _bspec.loader.exec_module(_basins)
    for _b in _basins.BASINS:
        _nm = _basins.model_names(_b)
        mf_path = DATA_DIR / _nm["forecast"]
        if mf_path.exists():
            mf = load_json(mf_path)
            if not mf.get("degraded"):
                walk(mf, f"model_forecast[{_b}]")
                for h in ("h1", "h2"):
                    if mf["predictions"][h]["q_m3s"].get("class") != "FORECAST":
                        errors.append(f"model_forecast[{_b}].{h}: prediction must be FORECAST")
                if mf["colour"].get("class") != "SIMULATED":
                    errors.append(f"model_forecast[{_b}].colour: decision rule must be SIMULATED")
            n_extra.append(_nm["forecast"] + (" (degraded)" if mf.get("degraded") else ""))
        mt_path = PUBLIC_DIR / _nm["test"]
        if mt_path.exists():
            mt = load_json(mt_path)
            walk(mt, f"model_2026_test[{_b}]")
            for rec in mt.get("predicted_h1_m3s", []):
                if rec.get("class") != "FORECAST" or rec.get("replay_test") is not True:
                    errors.append(f"model_2026_test[{_b}]: predictions must be FORECAST + replay_test")
                    break
            for rec in mt.get("actual_m3s", []):
                if rec.get("class") != "OBSERVED":
                    errors.append(f"model_2026_test[{_b}]: actuals must be OBSERVED")
                    break
            n_extra.append(_nm["test"])
        dr_path = PUBLIC_DIR / _nm["drift"]
        if dr_path.exists():
            walk(load_json(dr_path), f"model_drift[{_b}]")

    # Track T2: CWC official gauges payload (public flood dissemination feed)
    cwc_path = PUBLIC_DIR / "cwc_stations.json"
    if cwc_path.exists():
        cwc = load_json(cwc_path)
        walk(cwc, "cwc_stations")
        if "Central Water Commission" not in cwc.get("attribution", ""):
            errors.append("cwc_stations: CWC attribution missing")
        for s in cwc.get("stations", []):
            if s.get("degraded"):
                continue
            if s["observed_level_m"].get("class") != "OBSERVED":
                errors.append(f"cwc_stations {s['aff_station']}: observed level must be OBSERVED")
            if s["cwc_forecast_peak_m"].get("class") != "FORECAST":
                errors.append(f"cwc_stations {s['aff_station']}: CWC forecast must be FORECAST")
            for k in ("warning_level_m", "danger_level_m"):
                if s[k].get("value") is None:
                    errors.append(f"cwc_stations {s['aff_station']}: {k} missing")
        n_extra.append(f"cwc_stations.json({len(cwc.get('stations', []))} stations, "
                       f"{len(cwc.get('degraded_stations', []))} degraded)")

    # SACHET official alerts payload (NDMA CAP public feed)
    sa_path = PUBLIC_DIR / "sachet_alerts.json"
    if sa_path.exists():
        sa = load_json(sa_path)
        walk(sa, "sachet_alerts")
        if "SACHET" not in sa.get("attribution", "") and "NDMA" not in sa.get("attribution", ""):
            errors.append("sachet_alerts: NDMA/SACHET attribution missing")
        for a in sa.get("active_alerts", []):
            for k in ("headline", "severity", "expires"):
                if a.get(k, {}).get("class") != "OBSERVED":
                    errors.append(f"sachet_alerts {a.get('identifier')}: {k} must be OBSERVED")
        n_extra.append(f"sachet_alerts.json({len(sa.get('active_alerts', []))} active)")

    # Phase 1: static-history manifest must match the bytes on disk
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("manifest", DATA_DIR.parent / "backend" / "model" / "manifest.py")
    _man = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_man)
    errors.extend(_man.verify())
    n_extra.append("MANIFEST.json")

    # Phase 1: label store + drift payload
    lab = DATA_DIR / "labels" / "events.csv"
    if lab.exists():
        side = DATA_DIR / "labels" / "events.provenance.json"
        if not side.exists():
            errors.append("labels: events.provenance.json missing")
        else:
            sc = load_json(side)
            if sc.get("class") != "OBSERVED" or not sc.get("retrieved_at"):
                errors.append("labels: sidecar incomplete")
        n_extra.append("labels/events.csv")

    # Forecast scoreboard payload
    sb_path = PUBLIC_DIR / "forecast_scoreboard.json"
    if sb_path.exists():
        walk(load_json(sb_path), "forecast_scoreboard")
        n_extra.append("forecast_scoreboard.json")

    # Model v0 history partitions: every parquet needs a provenance sidecar
    hist = DATA_DIR / "history"
    if hist.exists():
        n_parts = 0
        for pq in hist.glob("*/*/*.parquet"):
            side = pq.with_name(pq.stem + ".provenance.json")
            if pq.parent.parent.name == "glofas" and pq.stem == "forecasts":
                pass  # archived forecasts: sidecar class FORECAST + archived:true (accepted below)
            if not side.exists():
                errors.append(f"history {pq.parent.name}/{pq.name}: sidecar missing")
                continue
            sc = load_json(side)
            # rainfall/discharge history is OBSERVED reanalysis; rainfall_fc
            # partitions are archived ISSUED forecasts -> class FORECAST
            # with archived:true (they are predictions, retrieved later)
            cls_ok = (sc.get("class") == "OBSERVED"
                      or (sc.get("class") == "FORECAST" and sc.get("archived") is True))
            if not cls_ok or not sc.get("source") or not sc.get("retrieved_at"):
                errors.append(f"history {pq.parent.name}/{pq.name}: sidecar incomplete")
            if pq.stem == "2026" and not sc.get("partial_test_only"):
                errors.append(f"history {pq.parent.name}/2026: must be flagged partial_test_only")
            n_parts += 1
        if n_parts:
            n_extra.append(f"history({n_parts} partitions)")

    if errors:
        print(f"PROVENANCE CHECK FAILED - {len(errors)} problem(s):")
        for e in errors[:40]:
            print("  -", e)
        return 1
    print(f"OK provenance: {len(vs['villages'])} villages, {len(rs['rivers'])} rivers, "
          f"rainfall+discharge raw docs, v2 payloads ({', '.join(n_extra)}) - "
          f"every value carries source/retrieved_at/class")
    return 0


if __name__ == "__main__":
    sys.exit(main())
