#!/usr/bin/env python3
"""Download and extract a KITTI raw drive into `data/kitti_raw/`.

KITTI data is CC BY-NC-SA 3.0 (Geiger et al.) and is **never committed to this repo** —
this script is how anyone reproduces the dataset locally. See `ATTRIBUTION.md`.

    python3 scripts/fetch_kitti.py                     # default drive (0009, city, 447 frames)
    python3 scripts/fetch_kitti.py --drive 0005        # a shorter one (154 frames, ~0.6 GB)
    python3 scripts/fetch_kitti.py --tracklets         # + object labels (tiny XML, no re-download)
    python3 scripts/fetch_kitti.py --list              # sizes, without downloading
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "kitti_raw"

# Drives worth starting from, all from the 2011_09_26 calibration day.
DRIVES = {
    "0005": "city, 154 frames — smallest useful drive, good for fast iteration",
    "0009": "city, 447 frames — default: long enough for real trajectory drift",
    "0027": "road, 188 frames — higher speed, tests the braking shield harder",
}


def _url(date: str, drive: str) -> str:
    name = f"{date}_drive_{drive}"
    return f"{BASE}/{name}/{name}_sync.zip"


def _calib_url(date: str) -> str:
    return f"{BASE}/{date}_calib.zip"


def _tracklets_url(date: str, drive: str) -> str:
    name = f"{date}_drive_{drive}"
    return f"{BASE}/{name}/{name}_tracklets.zip"


def remote_size(url: str) -> int | None:
    """Content-length of `url` in bytes, or None if the server won't say."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            return int(r.headers["Content-Length"])
    except Exception:
        return None


def download(url: str, dest: Path) -> None:
    """Fetch `url` to `dest` with a progress bar, resuming nothing (curl handles retries)."""
    if shutil.which("curl"):
        subprocess.run(["curl", "-#", "-fL", "-o", str(dest), url], check=True)
    else:
        urllib.request.urlretrieve(url, dest)


def fetch(date: str, drive: str, keep_zip: bool) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / date / f"{date}_drive_{drive}_sync"
    if out.exists():
        print(f"already present: {out}")
        return out

    for url in (_calib_url(date), _url(date, drive)):
        zip_path = DATA_DIR / Path(url).name
        size = remote_size(url)
        pretty = f"{size / 2**30:.2f} GB" if size else "unknown size"
        print(f"downloading {Path(url).name} ({pretty})")
        download(url, zip_path)

        print(f"extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
        if not keep_zip:
            zip_path.unlink()

    print(f"ready: {out}")
    return out


def fetch_tracklets(date: str, drive: str, keep_zip: bool) -> Path:
    """Download just the object labels (`tracklet_labels.xml`) — a few KB, not the whole drive.

    They ship separately from `_sync.zip`, so this can add labels to a drive already on disk
    without re-fetching gigabytes. Not every drive is labelled; a 404 here means this one is
    not (the S3 bucket returns HTML, and the zip extract then fails with a clear error).
    """
    out = DATA_DIR / date / f"{date}_drive_{drive}_sync" / "tracklet_labels.xml"
    if out.exists():
        print(f"already present: {out}")
        return out

    url = _tracklets_url(date, drive)
    zip_path = DATA_DIR / Path(url).name
    print(f"downloading {Path(url).name} (object labels)")
    download(url, zip_path)
    print(f"extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    if not keep_zip:
        zip_path.unlink()
    print(f"ready: {out}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26", help="KITTI calibration day")
    p.add_argument("--drive", default="0009", help=f"drive id; suggested: {', '.join(DRIVES)}")
    p.add_argument("--keep-zip", action="store_true", help="don't delete archives after extract")
    p.add_argument("--tracklets", action="store_true",
                   help="also fetch object labels (tiny; adds to a drive already on disk)")
    p.add_argument("--list", action="store_true", help="show suggested drives and exit")
    args = p.parse_args()

    if args.list:
        print("suggested drives (2011_09_26):")
        for d, desc in DRIVES.items():
            size = remote_size(_url("2011_09_26", d))
            pretty = f"{size / 2**30:.2f} GB" if size else "?"
            print(f"  {d}  {pretty:>9}  {desc}")
        return 0

    fetch(args.date, args.drive, args.keep_zip)
    if args.tracklets:
        fetch_tracklets(args.date, args.drive, args.keep_zip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
