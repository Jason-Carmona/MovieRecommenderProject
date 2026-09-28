#!/usr/bin/env python3
"""Sweep shrinkage (lambda) and neighbourhood size (K).

    python scripts/sweep.py --data data/raw/ml-32m

Both knobs were left at first-guess defaults (25 and 100) while the pipeline
was being built. This finds out what they should be.

The grid is fitted once per lambda, not once per cell: `ItemItemCF.truncate`
derives every smaller K from the largest one, because the 50 best neighbours of
an item are a subset of its 200 best. Fitting is ~3.5 minutes on ml-32m and
truncation is seconds, so this turns a 51-minute sweep into a 17-minute one.
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
    PopularityRecommender, compare, evaluate, temporal_split,
)
from lbxd.ratings import load_ratings                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--shrinkage", type=float, nargs="*",
                    default=[0.0, 10.0, 25.0, 50.0, 100.0])
    ap.add_argument("--top-k", type=int, nargs="*", default=[50, 100, 200])
    ap.add_argument("--damping", type=float, default=0.15)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--max-users", type=int, default=2000)
    ap.add_argument("--min-item-ratings", type=int, default=5)
    args = ap.parse_args()

    raw = (load_ratings(args.data, args.min_item_ratings) if args.data
           else synthetic.make_ratings())
    split = temporal_split(raw, max_users=args.max_users or None)
    print(f"{split.train}\n{split}\n")

    baseline = evaluate(PopularityRecommender(split.train), split, "popularity", k=args.k)
    print(f"{baseline}\n")

    largest = max(args.top_k)
    rows: list[tuple[float, int, float, float, int]] = []

    for shrinkage in args.shrinkage:
        started = time.perf_counter()
        full = ItemItemCF.fit(
            split.train.matrix, split.train.item_ids,
            top_k=largest, shrinkage=shrinkage,
        )
        print(f"lambda={shrinkage:g}: fitted at K={largest} "
              f"in {time.perf_counter() - started:.0f}s")

        for top_k in sorted(args.top_k):
            model = full if top_k == largest else full.truncate(top_k)
            result = evaluate(
                model, split, f"L={shrinkage:g} K={top_k}",
                k=args.k, popularity_damping=args.damping,
            )
            rows.append((shrinkage, top_k, result.ndcg, result.coverage,
                         model.similarity.nnz))
            print(f"    K={top_k:<4} NDCG@{args.k}={result.ndcg:.4f}  "
                  f"cov={result.coverage:.3f}  edges={model.similarity.nnz:,}")

    # Sorting a table by a noisy metric will always produce a "winner". Compare
    # every config against the incumbent *paired* on the same users, so the
    # table says which differences are real rather than which were luckiest.
    rows.sort(key=lambda r: -r[2])
    print(f"\n{'lambda':>7} {'K':>5} {'NDCG':>8} {'cov':>7} {'edges':>12} {'vs pop':>9}")
    for shrinkage, top_k, ndcg, cov, nnz in rows:
        lift = (ndcg - baseline.ndcg) / baseline.ndcg if baseline.ndcg else 0.0
        print(f"{shrinkage:>7g} {top_k:>5} {ndcg:>8.4f} {cov:>7.3f} "
              f"{nnz:>12,} {lift:>+8.1%}")

    best = rows[0]
    print(f"\nbest: lambda={best[0]:g} K={best[1]} -> NDCG@{args.k}={best[2]:.4f}")
    if best[0] == max(args.shrinkage) or best[1] == max(args.top_k):
        print("!! the winner sits on the edge of the grid — extend it before "
              "believing this is the optimum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
