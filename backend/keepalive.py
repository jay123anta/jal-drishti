"""Cross-platform keep-alive loop (alternative to the Windows Scheduled Task).

    python backend/keepalive.py            # run the pipeline every 3 h, forever
    python backend/keepalive.py --hours 6

Leaves a console window open; use scripts/install_keepalive.ps1 on Windows
for something that survives logout/sleep. Each run also calls
scripts/commit_archive.py (daily archive checkpoint).
"""

import datetime
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    hours = 3.0
    if "--hours" in sys.argv:
        hours = float(sys.argv[sys.argv.index("--hours") + 1])
    while True:
        t0 = time.monotonic()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"\n===== keepalive run {stamp} =====", flush=True)
        rc = subprocess.call([sys.executable, str(REPO / "backend" / "run_pipeline.py")])
        if rc == 0:
            subprocess.call([sys.executable, str(REPO / "scripts" / "commit_archive.py")])
        wait = max(hours * 3600 - (time.monotonic() - t0), 60)
        print(f"===== exit {rc}; next run in {wait / 3600:.1f} h (Ctrl+C to stop) =====", flush=True)
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            print("keepalive stopped")
            return 0


if __name__ == "__main__":
    sys.exit(main())
