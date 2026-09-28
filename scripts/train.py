#!/usr/bin/env python3
"""Fit the item-item model and write it to data/processed/model.npz.

    python scripts/train.py --data data/raw/ml-32m

Training is a batch job: run it once, ship the artifact, and serve
recommendations from it. Nothing at request time touches the ratings matrix.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbxd import synthetic                                        # noqa: E402
from lbxd.cf import ItemItemCF                                    # noqa: E402
from lbxd.ratings import load_ratings                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, help="MovieLens directory (default: synthetic)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/model.npz")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--shrinkage", type=float, default=100.0)
    ap.add_argument("--min-item-ratings", type=int, default=5)
    ap.add_argument("--block", type=int, default=512,
                    help="items per chunk; lower it if training runs out of memory")
    args = ap.parse_args()

    raw = (load_ratings(args.data, args.min_item_ratings) if args.data
           else synthetic.make_ratings())
    data = raw.to_dataset()
    print(data)

    started = time.perf_counter()
    model = ItemItemCF.fit(
        data.matrix, data.item_ids,
        top_k=args.top_k, shrinkage=args.shrinkage, block=args.block,
    )
    print(f"trained in {time.perf_counter() - started:.1f}s, "
          f"{model.similarity.nnz} similarity edges")

    model.save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
