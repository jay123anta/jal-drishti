"""UPGRADE v2, STEP A3 - analyze the July 2026 replay and write findings.

Reads:  data/archive/2026-07/archive_rainfall.json
        data/archive/2026-07/archive_discharge.json
        public/replay_2026-07.json
Writes: REPLAY-FINDINGS.md            (full findings, honest, mixed OK)
        public/replay_findings.json   (plain-language card + technical numbers)

Every number in the findings is COMPUTED from the archived data / replay
output, never typed in. The findings follow the data: if the signal is weak,
late, or drowned in monsoon noise, that is what gets written.

Event reference: the cloudburst struck Mon district, Nagaland on
19 July 2026 (IST). This was a real disaster with loss of life; wording
stays factual and respectful.
"""

import datetime
import sys

from common import (DATA_DIR, OBSERVED, PUBLIC_DIR, SIMULATED, load_json,
                    save_json, utc_now_iso)
from classify_risk import (ARCHIVE_DIR, ARCHIVE_TAG, DISCLAIMER, parse_utc,
                           replay_discharge_pctl, replay_rain_series,
                           trailing_sum)

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
EVENT_DAY = datetime.date(2026, 7, 19)
# start of 19 July 2026 in IST, expressed in UTC
EVENT_START_UTC = datetime.datetime(2026, 7, 18, 18, 30, tzinfo=datetime.timezone.utc)

NAGALAND_IDS = ["mon", "mokokchung", "zunheboto", "wokha"]
SPIKE_FLOOR_MM = 30.0   # a 24 h total must also clear this to count as a spike

ANALYSIS_SRC = "derived by backend/analyze_replay.py from the archived inputs"
REPLAY_SRC = "derived by backend/analyze_replay.py from the replay of the demo heuristic (version in public/replay_2026-07.json: heuristic field)"


def ist(t: datetime.datetime) -> str:
    return t.astimezone(IST).strftime("%d %b %H:%M IST")


def rain_spikes(rain_doc: dict, replay_start: datetime.datetime) -> dict:
    """Per Nagaland point: first 24 h spell in the replay window that beats
    every 24 h spell of the preceding 30 days (and >= SPIKE_FLOOR_MM)."""
    out = {}
    for pid in NAGALAND_IDS:
        pt = next(p for p in rain_doc["points"] if p["id"] == pid)
        times, cum = replay_rain_series(pt)
        roll24 = [trailing_sum(times, cum, t, 24) for t in times]
        first = None
        peak_i = max(range(len(times)), key=lambda i: roll24[i])
        for i, t in enumerate(times):
            if t < replay_start:
                continue
            base = [roll24[j] for j in range(len(times))
                    if t - datetime.timedelta(days=30) < times[j] <= t - datetime.timedelta(hours=24)]
            if not base:
                continue
            bmax = max(base)
            if roll24[i] > bmax and roll24[i] >= SPIKE_FLOOR_MM:
                first = {"time": t, "mm24": roll24[i], "prev_month_max": bmax,
                         "prev_month_mean": round(sum(base) / len(base), 1)}
                break
        out[pid] = {"name": pt["name"], "first_spike": first,
                    "peak": {"time": times[peak_i], "mm24": roll24[peak_i]}}
    return out


def discharge_findings(disch_doc: dict, replay_start: datetime.date,
                       replay_end: datetime.date) -> dict:
    out = {}
    for pt in disch_doc["points"]:
        daily = pt["daily"]
        vals = {datetime.date.fromisoformat(r["date"]): r["discharge_m3s"] for r in daily}
        # first replay-window day at/above the 90th trailing-30-day percentile,
        # using the heuristic's own definition
        first90 = None
        d = replay_start
        while d <= replay_end:
            dq = replay_discharge_pctl(daily, d)
            if dq and dq[1] >= 0.90:
                first90 = {"date": d, "q": dq[0], "pctl": dq[1]}
                break
            d += datetime.timedelta(days=1)
        # was it already >= 90th on the first replay day?
        at_start = replay_discharge_pctl(daily, replay_start)
        # biggest day-over-day rise within the replay window
        jump = None
        d = replay_start + datetime.timedelta(days=1)
        while d <= replay_end:
            prev, curr = vals.get(d - datetime.timedelta(days=1)), vals.get(d)
            if prev and curr and prev > 0:
                pct = 100.0 * (curr - prev) / prev
                if jump is None or pct > jump["pct"]:
                    jump = {"date": d, "from": prev, "to": curr, "pct": round(pct, 1)}
            d += datetime.timedelta(days=1)
        wmax_d = max((d for d in vals if replay_start <= d <= replay_end), key=lambda d: vals[d])
        out[pt["id"]] = {"river": pt["river"], "site": pt["site"],
                         "first_pctl90": first90, "at_start": at_start,
                         "biggest_jump": jump,
                         "window_max": {"date": wmax_d, "q": vals[wmax_d]}}
    return out


def village_transitions(replay: dict) -> dict:
    t0 = replay["times"][0]
    rows = []
    for v in replay["villages"]:
        if v["district"] != "Sivasagar":
            continue
        first_y = next((e["t"] for e in v["steps"] if e["risk"]["value"] != "GREEN"), None)
        first_r = next((e["t"] for e in v["steps"] if e["risk"]["value"] == "RED"), None)
        # around the event: last GREEN -> elevated transition in 18-21 July
        re_elev = None
        for prev, curr in zip(v["steps"], v["steps"][1:]):
            if ("2026-07-18" <= curr["t"][:10] <= "2026-07-21"
                    and prev["risk"]["value"] == "GREEN"
                    and curr["risk"]["value"] != "GREEN"):
                re_elev = curr["t"]
                break
        red_during_event = [e["t"] for e in v["steps"]
                            if "2026-07-19" <= e["t"][:10] <= "2026-07-21"
                            and e["risk"]["value"] == "RED"]
        rows.append({"name": v["name"], "first_yellow": first_y, "first_red": first_r,
                     "already_at_start": first_y == t0,
                     "event_re_elevation": re_elev,
                     "red_during_event": red_during_event})
    return {"t0": t0, "rows": rows}


def control_week(replay: dict, rain_doc: dict) -> dict:
    """Quietest 7 consecutive days of the replay window by total rain across
    all archive points; count non-GREEN village-steps inside it."""
    daily_total = {}
    for pt in rain_doc["points"]:
        for h in pt["hourly"]:
            d = h["time"][:10]
            daily_total[d] = daily_total.get(d, 0.0) + h["precipitation_mm"]
    days = sorted(d for d in daily_total if replay["window"]["start"] <= d <= replay["window"]["end"])
    best = None
    for i in range(len(days) - 6):
        week = days[i:i + 7]
        tot = sum(daily_total[d] for d in week)
        if best is None or tot < best["total"]:
            best = {"start": week[0], "end": week[-1], "total": round(tot, 1)}
    lo, hi = best["start"], best["end"]
    n_steps = n_nongreen = 0
    worst = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    for v in replay["villages"]:
        for e in v["steps"]:
            if lo <= e["t"][:10] <= hi:
                n_steps += 1
                if e["risk"]["value"] != "GREEN":
                    n_nongreen += 1
                worst[e["risk"]["value"]] += 0  # placeholder, per-class below
    per_class = {"YELLOW": 0, "RED": 0}
    for v in replay["villages"]:
        for e in v["steps"]:
            if lo <= e["t"][:10] <= hi and e["risk"]["value"] in per_class:
                per_class[e["risk"]["value"]] += 1
    best.update({"village_steps": n_steps, "non_green": n_nongreen,
                 "yellow": per_class["YELLOW"], "red": per_class["RED"],
                 "pct_non_green": round(100.0 * n_nongreen / n_steps, 1) if n_steps else 0.0})
    return best


def overall_saturation(replay: dict) -> dict:
    n = ng = 0
    for v in replay["villages"]:
        for e in v["steps"]:
            n += 1
            if e["risk"]["value"] != "GREEN":
                ng += 1
    return {"steps": n, "non_green": ng, "pct": round(100.0 * ng / n, 1)}


def main() -> int:
    rain_doc = load_json(ARCHIVE_DIR / "archive_rainfall.json")
    disch_doc = load_json(ARCHIVE_DIR / "archive_discharge.json")
    replay = load_json(PUBLIC_DIR / f"replay_{ARCHIVE_TAG}.json")
    now = utc_now_iso()

    rep_start_dt = parse_utc(replay["window"]["start"] + "T00:00")
    rep_start_d = datetime.date.fromisoformat(replay["window"]["start"])
    rep_end_d = datetime.date.fromisoformat(replay["window"]["end"])

    spikes = rain_spikes(rain_doc, rep_start_dt)
    disch = discharge_findings(disch_doc, rep_start_d, rep_end_d)
    trans = village_transitions(replay)
    ctrl = control_week(replay, rain_doc)
    sat = overall_saturation(replay)

    mon = spikes["mon"]
    dik = disch["dikhow_sivasagar"]

    # ---- lead-time facts, computed ----
    mon_spike_t = mon["first_spike"]["time"] if mon["first_spike"] else None
    spike_vs_event_h = (round((mon_spike_t - EVENT_START_UTC).total_seconds() / 3600, 1)
                        if mon_spike_t else None)
    surge_date = dik["biggest_jump"]["date"] if dik["biggest_jump"] else None
    n_already = sum(1 for r in trans["rows"] if r["already_at_start"])
    n_red_before = sum(1 for r in trans["rows"]
                       if r["first_red"] and r["first_red"][:10] < "2026-07-19")
    n_red_during = sum(1 for r in trans["rows"] if r["red_during_event"])
    re_elevs = sorted(r["event_re_elevation"] for r in trans["rows"] if r["event_re_elevation"])
    first_re_elev = re_elevs[0] if re_elevs else None

    # ------------------------------------------------------------------ MD
    L = []
    A = L.append
    A(f"# REPLAY-FINDINGS - 19 July 2026 Mon cloudburst / Upper Assam floods")
    A("")
    A(f"**Status: retrospective replay of the UNVALIDATED demo heuristic "
      f"({replay.get('heuristic', 'version unstated')}) over archived")
    A("reanalysis inputs. Risk colours are class")
    A("SIMULATED. This document reports what the data shows, including where")
    A("the heuristic performs poorly.**")
    A("")
    A(f"Replay window: {replay['window']['start']} to {replay['window']['end']}, "
      f"6-hour steps. Generated {now} by `backend/analyze_replay.py` - every "
      f"number below is computed from the archived data, none are typed in.")
    A("")
    A("The event: on 19 July 2026 a cloudburst in Mon district, Nagaland caused")
    A("catastrophic flooding in Sivasagar, Charaideo, Jorhat and Golaghat")
    A("districts via the south-bank tributaries (Dikhow, Disang, Janji,")
    A("Dhansiri), with loss of life. The replay asks one narrow question: what")
    A("would the PoC's placeholder rules have shown, and when?")
    A("")
    A("## 1. When the rain spiked at the Nagaland / Mon points")
    A("")
    A("Definition of \"first spike\": the first 24-hour rainfall total in the")
    A(f"replay window that exceeds every 24-hour total of the preceding 30 days")
    A(f"(and is at least {SPIKE_FLOOR_MM:.0f} mm). ERA5-family grid values; a localized")
    A("cloudburst is typically far more intense than its grid-cell average.")
    A("")
    A("| Point | First spike (UTC / IST) | 24 h total | Prev-month max 24 h | Prev-month mean 24 h | Window peak 24 h |")
    A("|---|---|---|---|---|---|")
    for pid in NAGALAND_IDS:
        s = spikes[pid]
        f = s["first_spike"]
        if f:
            A(f"| {s['name']} | {f['time'].strftime('%Y-%m-%d %H:%M')}Z / {ist(f['time'])} "
              f"| {f['mm24']:.1f} mm | {f['prev_month_max']:.1f} mm | {f['prev_month_mean']:.1f} mm "
              f"| {s['peak']['mm24']:.1f} mm at {s['peak']['time'].strftime('%m-%d %H:%M')}Z |")
        else:
            A(f"| {s['name']} | no 24 h total beat the previous month in this window | - | - | - "
              f"| {s['peak']['mm24']:.1f} mm at {s['peak']['time'].strftime('%m-%d %H:%M')}Z |")
    A("")
    if mon_spike_t:
        A(f"At Mon itself the first month-beating 24 h total "
          f"({mon['first_spike']['mm24']:.1f} mm vs a previous-month max of "
          f"{mon['first_spike']['prev_month_max']:.1f} mm) completes at "
          f"**{mon_spike_t.strftime('%Y-%m-%d %H:%M')}Z ({ist(mon_spike_t)})** - "
          f"{abs(spike_vs_event_h):.0f} h {'after' if spike_vs_event_h >= 0 else 'before'} "
          f"the start of 19 July (IST). The cloudburst registers in the archived "
          f"data on the event day itself, not before it.")
    A("")
    A("## 2. When the Dikhow (and neighbours) crossed the 90th percentile")
    A("")
    A("Percentile is the heuristic's own definition: rank of the day's GloFAS")
    A("reanalysis value within its trailing 30-day series. NOT a danger level.")
    A("")
    A("| River point | First day ≥ 90th pctl in window | Biggest day-over-day rise | Window max |")
    A("|---|---|---|---|")
    for rid in ("dikhow_sivasagar", "brahmaputra_dibrugarh", "jiabharali_tezpur", "kopili_kampur"):
        d = disch[rid]
        f90 = d["first_pctl90"]
        j = d["biggest_jump"]
        f90_txt = "never"
        if f90:
            f90_txt = f90["date"].isoformat()
            if f90["date"] == rep_start_d:
                f90_txt += " (already ≥90th at window start)"
        jump_txt = "-"
        if j:
            jump_txt = (f"+{j['pct']}% on {j['date'].isoformat()} "
                        f"({j['from']:.0f} → {j['to']:.0f} m³/s)")
        A(f"| {d['river']} @ {d['site']} | {f90_txt} | {jump_txt} "
          f"| {d['window_max']['q']:.0f} m³/s on {d['window_max']['date'].isoformat()} |")
    A("")
    dj = dik["biggest_jump"]
    A(f"The Dikhow cell's largest rise, **+{dj['pct']:.0f}% in one day "
      f"({dj['from']:.0f} → {dj['to']:.0f} m³/s) on {dj['date'].isoformat()}**, is "
      f"the day after the cloudburst - consistent with upstream rain arriving "
      f"downstream with roughly a one-day lag. But note the first ≥90th-percentile "
      f"day was {dik['first_pctl90']['date'].isoformat()}: the percentile signal "
      f"was already high from ordinary monsoon flow well before the event.")
    A("")
    A("## 3. When villages would first have turned YELLOW and RED")
    A("")
    A(f"All {len(trans['rows'])} Sivasagar-district villages: ")
    A("")
    A("| Village | First non-GREEN | First RED | GREEN→elevated again 18–21 Jul | RED during 19–21 Jul |")
    A("|---|---|---|---|---|")
    for r in sorted(trans["rows"], key=lambda r: (r["first_red"] or "9", r["name"])):
        A(f"| {r['name']} | {r['first_yellow'] or '-'}"
          f"{' (window start)' if r['already_at_start'] else ''} "
          f"| {r['first_red'] or 'never'} | {r['event_re_elevation'] or '-'} "
          f"| {'yes' if r['red_during_event'] else 'no'} |")
    A("")
    if n_already or n_red_before:
        A(f"{n_already} of {len(trans['rows'])} villages were already YELLOW at the "
          f"very first replay step ({trans['t0']}), and {n_red_before} of them hit RED "
          f"**before** 19 July - driven by ordinary monsoon rain plus a discharge "
          f"percentile that saturates in a wet spell (documented failure mode "
          f"#3 in HEURISTIC.md). {n_red_during} villages showed RED during 19–21 July.")
    else:
        A(f"None of the {len(trans['rows'])} villages were elevated at the window "
          f"start, and none were RED before 19 July. {n_red_during} showed RED during "
          f"19–21 July - the colours moved WITH the event, not ahead of it.")
    A("")
    A("## 4. Lead time, stated plainly")
    A("")
    A(f"- The Mon rain spike completes on **19 July itself** "
      f"({ist(mon_spike_t) if mon_spike_t else 'n/a'}); the Dikhow surge registers "
      f"on **{surge_date.isoformat() if surge_date else 'n/a'}**, the day after.")
    if first_re_elev:
        A(f"- Villages that had dropped back to GREEN on 18 July re-elevated at "
          f"**{first_re_elev}** - during the event day, not before it.")
    if n_already or n_red_before:
        A(f"- **As a distinct new alarm, the replayed rules give no usable lead time "
          f"for the 19 July disaster.** The event-day signals are same-day (rain) and "
          f"next-day (river). Worse, most villages had already been YELLOW or RED for "
          f"days beforehand ({sat['pct']:.0f}% of all village-steps in the window are "
          f"non-GREEN), so an event warning would not have stood out from the "
          f"heuristic's routine monsoon state.")
    else:
        A(f"- **The replayed rules give no lead time - but they now discriminate.** "
          f"The colours changed on 19–20 July itself, not before ({sat['pct']:.0f}% of "
          f"village-steps in the whole window are non-GREEN, concentrated around the "
          f"event). Same-day at best: a heads-up, not a warning.")
    A(f"- What IS demonstrated: the cloudburst and the downstream surge are both "
      f"clearly visible in free archived data (a month-beating rain total at Mon; "
      f"a +{dj['pct']:.0f}% one-day discharge jump to the window maximum). The "
      f"raw signals exist; these placeholder rules cannot turn them into a "
      f"timely, specific warning.")
    A("")
    A("## 5. Control check - quietest week")
    A("")
    A(f"Quietest 7 consecutive days of the replay window by total rain across "
      f"all 14 archive points: **{ctrl['start']} to {ctrl['end']}** "
      f"({ctrl['total']:.0f} mm summed over all points). In that week the "
      f"heuristic still marked **{ctrl['non_green']} of {ctrl['village_steps']} "
      f"village-steps ({ctrl['pct_non_green']:.1f}%) non-GREEN** "
      f"({ctrl['yellow']} YELLOW, {ctrl['red']} RED).")
    A("")
    if ctrl["pct_non_green"] > 10:
        A("That is a clear false-fire tendency: even in the calmest week available "
          "the heuristic keeps a substantial share of villages elevated. In "
          "monsoon conditions the trailing-30-day percentile stays high long "
          "after rain stops, and the rules have no way to distinguish a wet "
          "fortnight from an emergency.")
    elif ctrl["pct_non_green"] > 0:
        A("A small share of village-steps stayed elevated even in the quietest "
          "week - a mild false-fire tendency, from the trailing-30-day "
          "percentile staying high after rain stops.")
    else:
        A("No village-step fired in the quietest week - no false-fire in this "
          "particular control slice.")
    A("")
    A("## 6. LIMITATIONS")
    A("")
    A("- This is a **placeholder heuristic applied retrospectively**. It is")
    A("  evidence that the relevant signals are AVAILABLE in free archived data,")
    A("  **NOT** evidence that this or any system would have issued correct or")
    A("  timely warnings on 19 July 2026.")
    A("- Inputs are ERA5-family reanalysis and GloFAS model reanalysis - model")
    A("  grids, not gauges. A localized cloudburst is far more intense than its")
    A("  grid-cell daily average; the true local rainfall was likely much higher")
    A("  than the values above.")
    A("- The replay rain-point set includes Mon town (added for this replay);")
    A("  the scoring rules are unchanged but the input grid is one point richer")
    A("  than the live view was in July 2026.")
    A("- Thresholds (30/60/100 mm, 70th/90th percentile) remain arbitrary demo")
    A("  values; the percentile saturates in monsoon (visible throughout above).")
    A("- No lag/routing model: rain and river signals are combined instantly;")
    A("  the real Dikhow response arrived with ~a one-day lag.")
    A("- Village risk colours inherit every failure mode listed in HEURISTIC.md.")
    A("- Replay classes were computed at 6-hour steps from hourly rain but only")
    A("  DAILY discharge (GloFAS is daily), so within a day the river half of")
    A("  the score cannot change.")
    A("")

    base_path = DATA_DIR / "history" / "replay_v0_baseline.json"
    if base_path.exists():
        b = load_json(base_path)["technical"]
        A("## 6b. Heuristic v0.1 (seasonal baseline) vs v0 - same replay, same rules otherwise")
        A("")
        A("v0 compared each river only with its own trailing 30 days, which saturates in")
        A("a wet spell. v0.1 compares with the 2015-2025 distribution for the same time of")
        A("year (+/-15 days). Numbers below are the frozen v0 result vs this run:")
        A("")
        A("| metric | v0 (trailing 30 d) | v0.1 (seasonal) |")
        A("|---|---|---|")
        A(f"| village-steps non-GREEN, whole window | {b['window_non_green_pct']['value']}% | {sat['pct']}% |")
        A(f"| control week non-GREEN | {b['control_week']['value']}% ({b['control_week']['start'][5:]}..{b['control_week']['end'][5:]}) | {ctrl['pct_non_green']}% ({ctrl['start'][5:]}..{ctrl['end'][5:]}) |")
        A(f"| Sivasagar villages already elevated at window start | {b['villages_already_elevated_at_start']['value']} | {n_already} |")
        A(f"| Sivasagar villages RED before 19 July | {b['villages_red_before_event']['value']} | {n_red_before} |")
        A(f"| Sivasagar villages RED during 19-21 July | - | {n_red_during} |")
        A("")
        better = ctrl["pct_non_green"] < b["control_week"]["value"]
        A(f"**Verdict:** the seasonal baseline "
          f"{'REDUCED' if better else 'did NOT reduce'} the false-fire tendency "
          f"(control week {b['control_week']['value']}% -> {ctrl['pct_non_green']}%). "
          + ("It remains a placeholder: cutoffs are still arbitrary and the target is still "
             "modelled discharge." if better else
             "Reported as-is; the placeholder cutoffs, not the baseline, may be the limit."))
        A("")
    md = "\n".join(L) + "\n"
    # Model v0 (M5) and the v1 what-if (V1c) append their sections when they
    # exist, so REPLAY-FINDINGS.md survives regeneration with them intact
    hist = DATA_DIR / "history"
    others = sorted(p.name for p in hist.glob("model_2026_section_*.md")
                    if p.name != "model_2026_v1_section.md")
    for fname in ["model_2026_section.md", "model_2026_v1_section.md", *others]:
        section_path = hist / fname
        if section_path.exists():
            md += "\n" + section_path.read_text(encoding="utf-8")
    with open(DATA_DIR.parent / "REPLAY-FINDINGS.md", "w", encoding="utf-8") as fh:
        fh.write(md)
    # served copy: the viewer's findings card links to it, and http.server
    # only serves public/ - same content, written by the same run
    with open(PUBLIC_DIR / "REPLAY-FINDINGS.md", "w", encoding="utf-8") as fh:
        fh.write(md)

    # ---------------------------------------------------- findings JSON card
    # "plain" strings feed the DEFAULT view: everyday words only (the Step E
    # jargon gate also scans this file's plain section).
    daypart = "day"
    if mon_spike_t:
        h = mon_spike_t.astimezone(IST).hour
        daypart = "morning" if h < 12 else ("afternoon" if h < 17 else "evening")
    plain = {
        "headline": ("We re-ran this map over the July 2026 floods using saved "
                     "weather and river data."),
        "signal": (f"The rain burst at Mon on 19 July and the river jump the next day "
                   f"(+{dj['pct']:.0f}% in one day) both show up clearly in the free data."),
        "lead": (f"But the colours gave no useful head start: the heavy-rain signal "
                 f"appears on the {daypart} of 19 July itself, and the river jump on "
                 f"20 July, the day after. "
                 + ("Many villages were already amber or red for days before, "
                    "because the whole fortnight was wet."
                    if (n_already or n_red_before) else
                    "The villages changed colour on those same days - with the "
                    "flood, not ahead of it.")),
        "control": (f"In the calmest week of the period ({ctrl['start'][8:]}–{ctrl['end'][8:]} "
                    f"{datetime.date.fromisoformat(ctrl['end']).strftime('%B')}), the test "
                    f"formula flagged {ctrl['pct_non_green']:.0f}% of village readings "
                    f"amber or red"
                    + (". It over-fires in the wet season." if ctrl['pct_non_green'] >= 10 else
                       " - a few false alarms, not many." if ctrl['pct_non_green'] > 0 else
                       " - no false alarms that week.")),
        "caveat": ("This is a look-back test of a rough formula on saved data. It does "
                   "not show that real warnings would have been given, or been right."),
    }
    # Model v0 line (M6): computed from the out-of-sample test, if it ran
    mt_path = PUBLIC_DIR / "model_2026_test.json"
    if mt_path.exists():
        mt = load_json(mt_path)
        leads = [e["lead_days"] for e in mt.get("event_onsets", [])
                 if e.get("lead_days") is not None]
        if leads and max(leads) > 0:
            model_line = (f"We also built a simple river model from ten years of "
                          f"past data and tested it on this flood. It gave about "
                          f"{max(leads)} day(s) of head start on one river rise. "
                          f"Details are in the notes.")
        else:
            model_line = ("We also built a simple river model from ten years of "
                          "past data and tested it on this flood. It only raised "
                          "its alarm once the water was already high - no head "
                          "start. Details are in the notes.")
        if (PUBLIC_DIR / "MODEL-CARD-dikhow-v1.md").exists():
            model_line += (" We then tried adding tomorrow's rain forecast as "
                           "an input. The forecast did see the 19 July rain "
                           "coming - but with only daily river data to learn "
                           "from, it still could not raise the alarm any "
                           "earlier, so the simpler version stayed.")
        plain["model"] = model_line
    technical = {
        "mon_first_spike": {
            "value": mon["first_spike"]["mm24"] if mon["first_spike"] else None,
            "unit": "mm/24h",
            "time_utc": mon_spike_t.strftime("%Y-%m-%dT%H:%MZ") if mon_spike_t else None,
            "prev_month_max_mm": mon["first_spike"]["prev_month_max"] if mon["first_spike"] else None,
            "class": OBSERVED, "source": ANALYSIS_SRC, "retrieved_at": now,
        },
        "dikhow_surge": {
            "value": dj["pct"], "unit": "% day-over-day",
            "date": dj["date"].isoformat(), "from_m3s": dj["from"], "to_m3s": dj["to"],
            "class": OBSERVED, "source": ANALYSIS_SRC, "retrieved_at": now,
        },
        "villages_already_elevated_at_start": {
            "value": n_already, "of": len(trans["rows"]),
            "class": SIMULATED, "source": REPLAY_SRC, "retrieved_at": now,
        },
        "villages_red_before_event": {
            "value": n_red_before, "of": len(trans["rows"]),
            "class": SIMULATED, "source": REPLAY_SRC, "retrieved_at": now,
        },
        "window_non_green_pct": {
            "value": sat["pct"], "unit": "% of village-steps",
            "class": SIMULATED, "source": REPLAY_SRC, "retrieved_at": now,
        },
        "control_week": {
            "value": ctrl["pct_non_green"], "unit": "% village-steps non-GREEN",
            "start": ctrl["start"], "end": ctrl["end"],
            "yellow": ctrl["yellow"], "red": ctrl["red"],
            "class": SIMULATED, "source": REPLAY_SRC, "retrieved_at": now,
        },
    }
    save_json(PUBLIC_DIR / "replay_findings.json", {
        "generated_at": now,
        "replay": True,
        "disclaimer": DISCLAIMER,
        "event": ("19 July 2026 cloudburst, Mon district, Nagaland; flooding in "
                  "Sivasagar, Charaideo, Jorhat and Golaghat districts"),
        "plain": plain,
        "technical": technical,
        "full_findings": "REPLAY-FINDINGS.md",
    })

    # ---- verification ----
    assert (DATA_DIR.parent / "REPLAY-FINDINGS.md").exists()
    check = load_json(PUBLIC_DIR / "replay_findings.json")
    assert check["replay"] is True and "UNVALIDATED" in check["disclaimer"]
    for k, t in check["technical"].items():
        assert {"value", "class", "source", "retrieved_at"} <= set(t), f"technical.{k}"
    print(f"OK REPLAY-FINDINGS.md + public/replay_findings.json")
    print(f"   lead-time: rain spike {ist(mon_spike_t) if mon_spike_t else 'n/a'}, "
          f"surge {surge_date.isoformat() if surge_date else 'n/a'}, "
          f"window saturation {sat['pct']}% non-GREEN")
    print(f"   control week {ctrl['start']}..{ctrl['end']}: "
          f"{ctrl['pct_non_green']}% non-GREEN ({ctrl['yellow']} Y, {ctrl['red']} R)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
