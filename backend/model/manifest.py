"""PHASE 1 (AI-ready platform) - dataset MANIFEST for the static history.

    python backend/model/manifest.py            # (re)write data/history/MANIFEST.json
    python backend/model/manifest.py --verify   # exit 1 on any mismatch

sha256 + size of every STATIC history partition and sidecar (rainfall,
discharge, rainfall_fc). The growing CWC archive (cwc_aff) is deliberately
excluded - it changes every fetch; its integrity is covered by its own
sidecars and archive_health. check_provenance.py calls verify() so a
silently modified training partition fails the pipeline.

The manifest's own hash is stamped into models/registry.json records, so a
metric can always be tied to the exact bytes it was computed from.
"""

import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import DATA_DIR, load_json, save_json, utc_now_iso  # noqa: E402

HIST = DATA_DIR / "history"
MANIFEST = HIST / "MANIFEST.json"
STATIC_DIRS = ["rainfall", "discharge", "rainfall_fc"]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan() -> dict:
    files = {}
    for d in STATIC_DIRS:
        base = HIST / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in (".parquet", ".json"):
                rel = p.relative_to(HIST).as_posix()
                files[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return files


def write() -> dict:
    files = scan()
    doc = {"generated_at": utc_now_iso(), "covers": STATIC_DIRS,
           "excluded": ["cwc_aff (growing archive; covered by its own sidecars + archive_health)"],
           "n_files": len(files), "files": files}
    save_json(MANIFEST, doc)
    return doc


def verify() -> list[str]:
    """Return a list of problems (empty = OK). Missing manifest = one problem."""
    if not MANIFEST.exists():
        return ["MANIFEST.json missing (run backend/model/manifest.py)"]
    doc = load_json(MANIFEST)
    now = scan()
    problems = []
    for rel, meta in doc["files"].items():
        if rel not in now:
            problems.append(f"manifest: {rel} missing on disk")
        elif now[rel]["sha256"] != meta["sha256"]:
            problems.append(f"manifest: {rel} CHANGED (hash mismatch)")
    for rel in now:
        if rel not in doc["files"]:
            problems.append(f"manifest: {rel} not in manifest (re-run manifest.py)")
    return problems


def main() -> int:
    if "--verify" in sys.argv:
        probs = verify()
        if probs:
            print(f"MANIFEST VERIFY FAILED - {len(probs)} problem(s):")
            for p in probs[:20]:
                print("  -", p)
            return 1
        print(f"OK manifest: {load_json(MANIFEST)['n_files']} static history files match")
        return 0
    doc = write()
    print(f"OK data/history/MANIFEST.json: {doc['n_files']} files hashed "
          f"({sum(f['bytes'] for f in doc['files'].values()) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
