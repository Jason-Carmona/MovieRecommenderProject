#!/usr/bin/env python3
"""Development launcher for the API.

    python scripts/serve.py --reload

Production runs `uvicorn lbxd.api:app` directly and configures everything by
environment variable. This script exists so a developer can point the service at
a different model or catalog without exporting anything, which is most of what
you do while building a front end against it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, help="model.npz from scripts/train.py")
    ap.add_argument("--catalog", type=Path, help="MovieLens directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument(
        "--origins",
        help="comma-separated front-end origins allowed to call this "
             "(default: localhost:3000 and localhost:5173)",
    )
    args = ap.parse_args()

    # The app reads its configuration from the environment, so the launcher's
    # job is just to populate it before the app is imported.
    if args.model:
        os.environ["MODEL_PATH"] = str(args.model)
    if args.catalog:
        os.environ["CATALOG_DIR"] = str(args.catalog)
    if args.origins:
        os.environ["ALLOWED_ORIGINS"] = args.origins

    import uvicorn

    print(f"docs on http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "lbxd.api:app",
        host=args.host, port=args.port, reload=args.reload,
        reload_dirs=[str(ROOT / "src")] if args.reload else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
