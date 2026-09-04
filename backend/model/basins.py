"""Basin configuration for the modelling recipe (history -> lag -> baselines).

Each basin: the PoC river point that is its TARGET (a verified snapped
GloFAS cell from data/discharge.json), the rainfall anchor cells over its
hill catchment, and upstream GloFAS candidates (kept only if the snap
probe finds a distinct plausible river cell - never forced).

Anchor coordinates are approximate town/area anchors; the value used is
always the reanalysis GRID CELL containing them (CATCHMENT.md). Catchment
mean = unweighted mean of the anchors (no drainage model; documented).
"""

BASINS = {
    "dikhow": {
        "label": "Dikhow (Naga Hills -> Sivasagar)",
        "target": "dikhow_sivasagar",
        "rain_points": [
            {"id": "zunheboto",   "lat": 26.01, "lon": 94.52},
            {"id": "satakha",     "lat": 26.06, "lon": 94.42},
            {"id": "aghunato",    "lat": 26.12, "lon": 94.63},
            {"id": "mokokchung",  "lat": 26.32, "lon": 94.52},
            {"id": "longkhim",    "lat": 26.44, "lon": 94.72},
            {"id": "changtongya", "lat": 26.46, "lon": 94.48},
            {"id": "naginimora",  "lat": 26.72, "lon": 94.85},
            {"id": "mon",         "lat": 26.75, "lon": 95.05},
        ],
        "upstream": [
            {"id": "dikhow_naginimora", "lat": 26.75, "lon": 94.80, "qmin": 20, "qmax": 2000},
            {"id": "dikhow_hills",      "lat": 26.35, "lon": 94.60, "qmin": 5,  "qmax": 1000},
        ],
        "cwc_station": "SIVASAGAR",
    },
    "kopili": {
        "label": "Kopili (Jaintia / Karbi Anglong hills -> Kampur)",
        "target": "kopili_kampur",
        "rain_points": [
            {"id": "jowai",        "lat": 25.45, "lon": 92.20},   # Jaintia Hills headwaters
            {"id": "khliehriat",   "lat": 25.36, "lon": 92.37},   # eastern Jaintia plateau
            {"id": "umrongso",     "lat": 25.45, "lon": 92.72},   # Kopili reservoir / gorge
            {"id": "hamren",       "lat": 25.70, "lon": 92.85},   # Karbi Anglong (existing PoC point)
            {"id": "baithalangso", "lat": 25.90, "lon": 92.60},   # lower hills, west
            {"id": "kheroni",      "lat": 25.93, "lon": 92.83},   # lower hills, east
        ],
        "upstream": [
            {"id": "kopili_umrongso", "lat": 25.55, "lon": 92.70, "qmin": 10, "qmax": 2000},
            {"id": "kopili_hills",    "lat": 25.85, "lon": 92.65, "qmin": 20, "qmax": 3000},
        ],
        "cwc_station": "KAMPUR",
    },
    "jiabharali": {
        "label": "Jia Bharali / Kameng (Arunachal -> Sonitpur)",
        "target": "jiabharali_tezpur",
        "rain_points": [
            {"id": "bhalukpong", "lat": 27.01, "lon": 92.64},   # hills exit (existing PoC point)
            {"id": "tenga",      "lat": 27.20, "lon": 92.55},   # Tenga valley
            {"id": "bomdila",    "lat": 27.26, "lon": 92.42},   # West Kameng ridge
            {"id": "dirang",     "lat": 27.35, "lon": 92.24},   # upper Kameng
            {"id": "seppa",      "lat": 27.30, "lon": 92.93},   # East Kameng
            {"id": "tawang",     "lat": 27.59, "lon": 91.87},   # far headwaters
        ],
        "upstream": [
            {"id": "kameng_bhalukpong", "lat": 27.02, "lon": 92.62, "qmin": 100, "qmax": 6000},
        ],
        "cwc_station": "NT ROAD CROSSING JIA-BHARALI",
    },
    "barak": {
        "label": "Barak (Manipur / Dima Hasao hills -> Silchar)",
        "target": "barak_silchar",
        "rain_points": [
            {"id": "tamenglong",    "lat": 24.99, "lon": 93.50},   # headwater hills (existing PoC point)
            {"id": "senapati",      "lat": 25.27, "lon": 94.02},   # far headwaters (Mao/Senapati)
            {"id": "jiribam",       "lat": 24.80, "lon": 93.12},   # middle reach (existing)
            {"id": "haflong",       "lat": 25.17, "lon": 93.02},   # north-slope Jatinga (existing)
            {"id": "churachandpur", "lat": 24.33, "lon": 93.68},   # southern flank
            {"id": "kolasib",       "lat": 24.22, "lon": 92.68},   # Mizoram flank
        ],
        "upstream": [
            {"id": "barak_fulertal", "lat": 24.75, "lon": 93.05, "qmin": 100, "qmax": 6000},
            {"id": "barak_manipur",  "lat": 24.65, "lon": 93.35, "qmin": 30,  "qmax": 3000},
        ],
        "cwc_station": "ANNAPURNA GHAT",
    },
    "beki": {
        "label": "Beki / Kurichhu (Bhutan hills -> Barpeta)",
        "target": "beki_barpeta",
        "rain_points": [
            {"id": "gelephu",  "lat": 26.87, "lon": 90.49},   # Bhutan foothills
            {"id": "sarpang",  "lat": 26.86, "lon": 90.27},
            {"id": "tsirang",  "lat": 27.02, "lon": 90.12},
            {"id": "trongsa",  "lat": 27.50, "lon": 90.51},   # mid-Bhutan
            {"id": "zhemgang", "lat": 27.21, "lon": 90.66},
            {"id": "mathanguri_hills", "lat": 26.85, "lon": 90.95},  # border gorge
        ],
        "upstream": [
            {"id": "beki_border", "lat": 26.80, "lon": 90.95, "qmin": 50, "qmax": 6000},
        ],
        "cwc_station": "BEKI ROAD BRIDGE",
    },
    "dhansiri": {
        "label": "Dhansiri South / Doyang (Dimapur & Naga hills -> Golaghat)",
        "target": "dhansiri_golaghat",
        "rain_points": [
            {"id": "dimapur",    "lat": 25.90, "lon": 93.73},
            {"id": "diphu",      "lat": 25.84, "lon": 93.43},   # Karbi flank
            {"id": "medziphema", "lat": 25.74, "lon": 93.87},
            {"id": "bokajan",    "lat": 26.02, "lon": 93.78},
            {"id": "wokha",      "lat": 26.10, "lon": 94.26},   # Doyang catchment (existing point)
        ],
        "upstream": [
            {"id": "dhansiri_dimapur", "lat": 25.90, "lon": 93.75, "qmin": 20, "qmax": 2500},
        ],
        "cwc_station": "GOLAGHAT",
    },
    "disang": {
        "label": "Disang (Patkai / Mon hills -> Nanglamoraghat)",
        "target": "disang_nanglamoraghat",
        "rain_points": [
            {"id": "mon",         "lat": 26.75, "lon": 95.05},   # divide with the Dikhow (existing)
            {"id": "longwa",      "lat": 26.74, "lon": 95.22},   # Patkai crest
            {"id": "borhat_hills","lat": 26.80, "lon": 95.15},
            {"id": "sapekhati",   "lat": 26.96, "lon": 95.13},   # foothills
            # Goal C anchor additions (retrain under the same pre-registered rule)
            {"id": "tizit",       "lat": 26.92, "lon": 95.02},   # northern Mon district
            {"id": "naginimora",  "lat": 26.72, "lon": 94.85},   # foothill rain (reused cell)
        ],
        "upstream": [
            {"id": "disang_upper", "lat": 26.85, "lon": 94.95, "qmin": 20, "qmax": 2000},
        ],
        "cwc_station": "NANGLAMORAGHAT",
    },
    # Brahmaputra mainstem reaches: the honest "catchment" is the river itself
    # upstream (half of Tibet cannot be sampled with a few rain anchors);
    # rain anchors are nearby valley/tributary cells for local context only.
    "bputra_dibrugarh": {
        "label": "Brahmaputra reach (Siang/upper Assam -> Dibrugarh)",
        "target": "brahmaputra_dibrugarh",
        "rain_points": [
            {"id": "pasighat", "lat": 28.07, "lon": 95.33},   # Siang exit
            {"id": "along",    "lat": 28.17, "lon": 94.80},   # Siang basin
            {"id": "roing",    "lat": 28.15, "lon": 95.85},   # Dibang
            {"id": "tezu",     "lat": 27.92, "lon": 96.17},   # Lohit
            # Goal C anchor additions (retrain under the same pre-registered rule)
            {"id": "yingkiong", "lat": 28.63, "lon": 95.02},   # upper Siang
            {"id": "anini",     "lat": 28.80, "lon": 95.90},   # Dibang headwaters
        ],
        "upstream": [
            {"id": "siang_pasighat", "lat": 28.06, "lon": 95.33, "qmin": 1500, "qmax": None},
        ],
        "cwc_station": "DIBRUGARH",
    },
    "bputra_tezpur": {
        "label": "Brahmaputra reach (Dibrugarh -> Tezpur)",
        "target": "brahmaputra_tezpur",
        "rain_points": [
            {"id": "sivasagar",  "lat": 26.98, "lon": 94.63},   # existing history
            {"id": "naginimora", "lat": 26.72, "lon": 94.85},
            {"id": "bhalukpong", "lat": 27.01, "lon": 92.64},
            {"id": "tenga",      "lat": 27.20, "lon": 92.55},
        ],
        "upstream": [
            {"id": "brahmaputra_dibrugarh", "existing": True},
            {"id": "dikhow_sivasagar",      "existing": True},
            {"id": "dhansiri_golaghat",     "existing": True},
            {"id": "jiabharali_tezpur",     "existing": True},
        ],
        "cwc_station": "TEZPUR",
    },
    "bputra_guwahati": {
        "label": "Brahmaputra reach (Tezpur -> Guwahati)",
        "target": "brahmaputra_guwahati",
        "rain_points": [
            {"id": "hamren",   "lat": 25.70, "lon": 92.85},     # existing history
            {"id": "umrongso", "lat": 25.45, "lon": 92.72},
            {"id": "jowai",    "lat": 25.45, "lon": 92.20},
            {"id": "kheroni",  "lat": 25.93, "lon": 92.83},
        ],
        "upstream": [
            {"id": "brahmaputra_tezpur", "existing": True},
            {"id": "kopili_kampur",      "existing": True},
        ],
        "cwc_station": "GUWAHATI(D.C.COURT)",
    },
    "subansiri": {
        "label": "Subansiri (Arunachal hills -> Badatighat, Lakhimpur)",
        "target": "subansiri_badatighat",
        "rain_points": [
            {"id": "daporijo",   "lat": 27.99, "lon": 94.22},   # upper Subansiri
            {"id": "ziro",       "lat": 27.54, "lon": 93.83},   # Apatani plateau (Kamla/Panior)
            {"id": "raga",       "lat": 27.51, "lon": 94.13},   # Kamla mid-basin
            {"id": "gerukamukh", "lat": 27.55, "lon": 94.36},   # foothill exit (Lower Subansiri dam site)
            {"id": "dumporijo",  "lat": 27.90, "lon": 94.45},   # eastern flank
        ],
        "upstream": [
            # matches CWC AFF station CHOULDHOWAGHAT (SUBANSIRI, 27.449/94.250)
            {"id": "subansiri_chouldhowa", "lat": 27.449, "lon": 94.250, "qmin": 300, "qmax": None},
        ],
        "cwc_station": "BADATIGHAT",
    },
    "manas": {
        "label": "Manas (Bhutan hills -> NH crossing, Bongaigaon)",
        "target": "manas_nhcrossing",
        "rain_points": [
            {"id": "trongsa",     "lat": 27.50, "lon": 90.51},   # Mangdechhu (reused from beki)
            {"id": "zhemgang",    "lat": 27.21, "lon": 90.66},   # reused from beki
            {"id": "panbang",     "lat": 26.87, "lon": 91.02},   # Manas confluence in Bhutan
            {"id": "mongar",      "lat": 27.27, "lon": 91.24},   # Drangmechhu (eastern branch)
            {"id": "pemagatshel", "lat": 27.03, "lon": 91.40},   # south-eastern flank
        ],
        "upstream": [
            {"id": "manas_mathanguri", "lat": 26.79, "lon": 90.97, "qmin": 100, "qmax": None},
        ],
        "cwc_station": "MANAS N H CROSSING",
    },
    "ranganadi": {
        "label": "Ranganadi / Panyor (Ziro plateau & dam -> NT Road crossing, Lakhimpur)",
        "target": "ranganadi_ntxing",
        "rain_points": [
            {"id": "ziro",     "lat": 27.54, "lon": 93.83},   # Apatani plateau headwaters (reused cell)
            {"id": "yazali",   "lat": 27.35, "lon": 93.84},   # Ranganadi dam reach
            {"id": "kimin",    "lat": 27.32, "lon": 93.96},   # foothill exit
            {"id": "joram",    "lat": 27.45, "lon": 93.70},   # western flank
        ],
        "upstream": [
            {"id": "ranganadi_yazali", "lat": 27.35, "lon": 93.85, "qmin": 5, "qmax": 1500},
        ],
        "cwc_station": "RANGANADI NT ROAD CROSSING",
    },
    "katakhal": {
        "label": "Katakhal / Dhaleswari (Mizoram Tlawng -> Matijuri, Hailakandi)",
        "target": "katakhal_matijuri",
        "rain_points": [
            {"id": "aizawl",   "lat": 23.73, "lon": 92.72},   # Tlawng headwaters
            {"id": "kolasib",  "lat": 24.22, "lon": 92.68},   # mid-valley (reused cell)
            {"id": "bairabi",  "lat": 24.19, "lon": 92.54},   # border exit
            {"id": "lala",     "lat": 24.55, "lon": 92.61},   # Hailakandi plains edge
        ],
        "upstream": [
            {"id": "dhaleswari_bairabi", "lat": 24.20, "lon": 92.55, "qmin": 5, "qmax": 1500},
        ],
        "cwc_station": "MATIJURI",
    },
    "sankosh": {
        "label": "Sankosh / Puna Tsang Chhu (Bhutan -> Golokganj, Dhubri)",
        "target": "sankosh_golokganj",
        "rain_points": [
            {"id": "punakha",  "lat": 27.59, "lon": 89.86},   # upper Puna Tsang
            {"id": "wangdue",  "lat": 27.49, "lon": 89.90},
            {"id": "dagana",   "lat": 27.07, "lon": 89.88},   # lower Bhutan reach
            {"id": "tsirang",  "lat": 27.02, "lon": 90.12},   # eastern flank (reused cell)
        ],
        "upstream": [
            {"id": "sankosh_border", "lat": 26.40, "lon": 89.85, "qmin": 50, "qmax": None},
        ],
        "cwc_station": "GOLOKGANJ",
    },
}

# The remaining PoC river cells get DISCHARGE-ONLY history (for the seasonal
# baseline of the heuristic); their coordinates come from data/discharge.json.
DISCHARGE_ONLY_CELLS = ["brahmaputra_dibrugarh", "brahmaputra_tezpur",
                        "brahmaputra_guwahati", "barak_silchar", "beki_barpeta",
                        "dhansiri_golaghat", "disang_nanglamoraghat"]


def basin_ids() -> list[str]:
    return list(BASINS)


def rain_point_ids(basin: str) -> list[str]:
    return [p["id"] for p in BASINS[basin]["rain_points"]]


def model_names(basin: str) -> dict:
    """Artifact names per basin; the Dikhow keeps its original v0 names."""
    d = basin == "dikhow"
    return {
        "meta": "dikhow_v0_meta.json" if d else f"{basin}_v0_meta.json",
        "pkl": "dikhow_v0.pkl" if d else f"{basin}_v0.pkl",
        "metrics": "model_metrics.json" if d else f"model_metrics_{basin}.json",
        "baselines": "baseline_metrics.json" if d else f"baseline_metrics_{basin}.json",
        "lag": "lag_summary.json" if d else f"lag_summary_{basin}.json",
        "card": "MODEL-CARD-dikhow-v0.md" if d else f"MODEL-CARD-{basin}-v0.md",
        "forecast": "model_forecast_dikhow.json" if d else f"model_forecast_{basin}.json",
        "test": "model_2026_test.json" if d else f"model_2026_test_{basin}.json",
        "test_png": "model_2026_test.png" if d else f"model_2026_test_{basin}.png",
        "section": "model_2026_section.md" if d else f"model_2026_section_{basin}.md",
        "drift": "model_drift.json" if d else f"model_drift_{basin}.json",
        "method": f"model-v0-{basin}",
    }
