"""Daily auto-commit of the growing CWC gauge archive (called by keepalive).

Commits ONLY the archive paths (data/history/cwc_aff, public/cwc_stations.json,
data/labels) and only if the last such commit is older than 24 h, so the
git history gets one archive checkpoint per day instead of one per run.
Never touches other modified files; skips if git is unavailable or a
merge/rebase is in progress. Exit 0 always (a failed commit must never
break the keep-alive loop).
"""

import datetime
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PATHS = ["data/history/cwc_aff", "data/history/glofas", "data/history/sachet",
         "data/history/nwdp", "data/history/model_fc", "data/labels", "data/history/scoreboard.json",
         "docs/FORECAST-SCOREBOARD.md",
         "public"]   # whole served site: the public GitHub Pages copy stays fresh
MIN_AGE_H = 2   # push-per-run: the public Pages site stays 3-hourly fresh


def git(*args) -> tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> int:
    rc, _ = git("rev-parse", "--is-inside-work-tree")
    if rc != 0:
        print("commit_archive: not a git repo, skipping")
        return 0
    if any((REPO / ".git" / f).exists() for f in ("MERGE_HEAD", "REBASE_HEAD", "rebase-merge")):
        print("commit_archive: merge/rebase in progress, skipping")
        return 0
    existing = [p for p in PATHS if (REPO / p).exists()]
    rc, out = git("log", "-1", "--format=%ct", "--", *existing)
    if rc == 0 and out.strip():
        age_h = (datetime.datetime.now(datetime.timezone.utc).timestamp() - int(out)) / 3600
        if age_h < MIN_AGE_H:
            print(f"commit_archive: last archive commit {age_h:.1f} h ago, skipping")
            return 0
    rc, out = git("status", "--porcelain", "--", *existing)
    if not out.strip():
        print("commit_archive: nothing to commit")
        return 0
    git("add", "--", *existing)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    rc, out = git("commit", "-q", "-m",
                  f"CWC gauge archive checkpoint {stamp}\n\n"
                  f"Automated commit by scripts/commit_archive.py (keep-alive, 3-hourly).")
    print("commit_archive:", "committed" if rc == 0 else f"commit failed: {out[:200]}")
    if rc == 0:
        _, remotes = git("remote")
        if "origin" in remotes.split():
            rc2, out2 = git("push", "origin", "HEAD")
            print("commit_archive:", "pushed to origin" if rc2 == 0
                  else f"push failed (kept local, will retry next run): {out2[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
