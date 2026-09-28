#!/usr/bin/env python3
"""Compare the CF model against the baselines.

    python scripts/evaluate.py                        # synthetic data
    python scripts/evaluate.py --data data/raw/ml-latest-small

Every number printed is an average over held-out users, so the comparison
between rows is the point — the absolute values are capped by how many films
each user liked in their held-out window.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbxd import synthetic                                        # noqa: E402
from lbxd.cf import ItemItemCF                                    # noqa: E402
from lbxd.evaluate import (                                       # noqa: E402
    PopularityRecommender, RandomRecommender, evaluate, temporal_split,
)
from lbxd.ratings import load_ratings                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, help="MovieLens directory (default: synthetic)")
    ap.add_argument("-k", type=int, default=10, help="list length to score")
    ap.add_argument("--top-k", type=int, default=200, help="neighbours kept per item")
    ap.add_argument("--shrinkage", type=float, default=100.0)
    ap.add_argument("--min-item-ratings", type=int, default=5)
    ap.add_argument("--max-users", type=int, default=2000,
                    help="evaluate a sample of held-out users (0 = all)")
    ap.add_argument(
        "--damping", type=float, nargs="*", default=[0.0, 0.1, 0.3],
        help="popularity damping values to sweep",
    )
    args = ap.parse_args()

    if args.data:
        raw = load_ratings(args.data, min_item_ratings=args.min_item_ratings)
        source = str(args.data)
    else:
        raw = synthetic.make_ratings()
        source = "synthetic (no MovieLens download needed)"

    split = temporal_split(raw, max_users=args.max_users or None)
    print(f"data: {source}")
    print(f"{split.train}\n{split}\n")

    started = time.perf_counter()
    model = ItemItemCF.fit(
        split.train.matrix, split.train.item_ids,
        top_k=args.top_k, shrinkage=args.shrinkage,
    )
    print(f"trained in {time.perf_counter() - started:.2f}s, "
          f"{model.similarity.nnz} similarity edges\n")

    results = [
        evaluate(RandomRecommender(split.train.n_items), split, "random", k=args.k),
        evaluate(PopularityRecommender(split.train), split, "popularity", k=args.k),
    ]
    for damping in args.damping:
        label = f"item-item CF (damp={damping:g})"
        results.append(
            evaluate(model, split, label, k=args.k, popularity_damping=damping)
        )

    for r in results:
        print(r)

    baseline = next(r for r in results if r.name == "popularity")
    best = max((r for r in results if r.name.startswith("item-item")),
               key=lambda r: r.ndcg)
    if baseline.ndcg > 0:
        lift = (best.ndcg - baseline.ndcg) / baseline.ndcg
        print(f"\nbest CF beats popularity by {lift:+.1%} NDCG@{args.k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
