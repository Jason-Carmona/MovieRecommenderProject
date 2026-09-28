#!/usr/bin/env python3
"""Download and unpack a MovieLens dataset into data/raw/.

    python scripts/fetch_movielens.py            # ml-latest-small, ~1 MB
    python scripts/fetch_movielens.py --full     # ml-32m, ~250 MB

Use the small set while building; train on the full one.
"""

from __future__ import annotations

import argparse
import shutil
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://files.grouplens.org/datasets/movielens"
DATASETS = {"small": "ml-latest-small", "full": "ml-32m"}
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url: str, dest: Path, insecure: bool) -> None:
    context = ssl._create_unverified_context() if insecure else None
    with urllib.request.urlopen(url, context=context, timeout=120) as r, \
            dest.open("wb") as out:
        shutil.copyfileobj(r, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="ml-32m instead of the small set")
    ap.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS verification (grouplens.org lets its certificate lapse "
             "periodically; only use this if that is the error you hit)",
    )
    args = ap.parse_args()

    name = DATASETS["full" if args.full else "small"]
    target = RAW / name
    if (target / "movies.csv").is_file():
        print(f"{target} already present")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / f"{name}.zip"
    url = f"{BASE}/{name}.zip"
    print(f"downloading {url}")
    try:
        download(url, archive, args.insecure)
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        print(f"download failed: {reason}", file=sys.stderr)
        if "certificate" in str(reason).lower():
            print(
                "grouplens.org's certificate has expired again. Either wait for "
                "them to renew it, download the zip in a browser into data/raw/, "
                "or re-run with --insecure.",
                file=sys.stderr,
            )
        return 1

    with zipfile.ZipFile(archive) as z:
        z.extractall(RAW)
    archive.unlink()
    print(f"unpacked to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
