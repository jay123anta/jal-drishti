"""Shared helpers for the JalDrishti PoC backend.

Provenance rules (non-negotiable, see README):
- every numeric value carries source, retrieved_at (ISO8601 UTC), and
  class in {OBSERVED, FORECAST, SIMULATED}.
- live API unreachable after 3 retries with backoff -> caller saves the
  attempted request, falls back to a clearly-labelled SIMULATED fixture.
"""

import json
import time
import datetime
import pathlib

import requests

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FIXTURES_DIR = DATA_DIR / "fixtures"
PUBLIC_DIR = REPO_ROOT / "public"

OBSERVED = "OBSERVED"
FORECAST = "FORECAST"
SIMULATED = "SIMULATED"


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_json(url: str, params: dict | None = None, retries: int = 3,
               backoff_s: float = 2.0, timeout_s: float = 30.0) -> dict:
    """GET a JSON API with retries + exponential backoff.

    Raises RuntimeError after `retries` failed attempts; the caller applies
    the SIMULATED-fixture fallback rule.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as err:  # noqa: BLE001 - deliberate catch-all for retry loop
            last_err = err
            if attempt < retries:
                time.sleep(backoff_s * (2 ** (attempt - 1)))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def save_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)


def load_json(path: pathlib.Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_failed_request(name: str, url: str, params: dict, error: str) -> pathlib.Path:
    """Record the exact request we attempted, per the fallback rules."""
    out = FIXTURES_DIR / f"FAILED_REQUEST_{name}.json"
    save_json(out, {
        "attempted_at": utc_now_iso(),
        "url": url,
        "params": params,
        "error": error,
        "note": "Live API unreachable after retries. See RETRY-LIVE.md to swap real data back in.",
    })
    return out
