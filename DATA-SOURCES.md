# Data sources, attribution, and takedown

Every number in this repository carries machine-readable provenance
(source, retrieval time, and whether it is measured, forecast, or
computer-estimated). This file is the human-readable summary.

| Source | What we use | Access route | Terms as understood |
|---|---|---|---|
| Open-Meteo (open-meteo.com) | live + historical weather-model rainfall; GloFAS river data relay | public API | free for non-commercial use, attribution given |
| Copernicus GloFAS (EU) | modelled river discharge, reanalysis + forecasts | via Open-Meteo Flood API | Copernicus open data |
| Copernicus Sentinel-1 (ESA/EU) | radar scenes, derived flood-water extents | CDSE (registered account) | Copernicus open data, attribution given |
| Central Water Commission - Advisory Flood Forecast portal (aff.india-water.gov.in) | official gauge levels, forecasts, warning/danger marks; archived 3-hourly | public dissemination text files | public dissemination service; archived here for research with attribution |
| NDMA SACHET (sachet.ndma.gov.in) | official disaster alerts (CAP), archived | public CAP/RSS feed | feed declares itself public domain |
| NWIC National Water Data Portal (nwdp.nwic.gov.in) | observed rain gauges, river levels, discharge (Assam, Arunachal, Meghalaya, Manipur; CWC Arunachal) | published open-data CSV resources | open data portal, attribution given |
| IMD Pune (imdpune.gov.in) | 0.25° gridded daily rainfall (north-east window extract) | the page's own download form | publicly downloadable; north-east extract archived for research with attribution; full-India raw files are NOT redistributed here (reproduction recipe recorded instead) |
| OpenStreetMap contributors | village locations, base map data | Overpass API / OpenFreeMap tiles | ODbL, attribution given |
| OpenFreeMap / OpenMapTiles | vector map tiles | public tile service | free service, attribution shown on the map |

## Archived copies

Some directories under `data/` contain archived snapshots of public
government dissemination feeds (CWC AFF readings, SACHET alerts, NWDP
datasets, IMD extracts). They are kept because the live feeds are
rolling windows - the public record disappears otherwise. They are
republished here solely for transparent, non-commercial flood research,
with full attribution, unmodified except for documented format
conversion.

**Takedown:** if you represent an originating agency and want any
archived data removed from this repository, open an issue or contact the
repository owner - it will be removed promptly and the pipeline adjusted
to keep such data local-only.

## Not a warning system

Everything here is a research demonstration. Model outputs are trained
and evaluated against modelled river data (GloFAS), not observed gauge
records, and colours on the map are estimates that can be wrong. Real
flood warnings come from ASDMA, CWC, and District Administrations -
the map itself displays their official alerts above its own estimates.
