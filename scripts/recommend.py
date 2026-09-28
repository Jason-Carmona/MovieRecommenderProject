#!/usr/bin/env python3
"""Recommend films for one Letterboxd export.

    python scripts/recommend.py ~/Downloads/letterboxd-export.zip

Expects a model from scripts/train.py and a catalog from the same MovieLens
directory it was trained on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbxd.cf import ItemItemCF                                    # noqa: E402
from lbxd.letterboxd import load_export                           # noqa: E402
from lbxd.matching import Tier, TitleIndex                        # noqa: E402
from lbxd.movielens import load_catalog                           # noqa: E402
from lbxd.recommend import recommend_for_profile                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export", type=Path, help="Letterboxd export zip or directory")
    ap.add_argument("--model", type=Path, default=ROOT / "data/processed/model.npz")
    ap.add_argument("--catalog", type=Path, default=ROOT / "data/raw/ml-latest-small")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--damping", type=float, default=0.15,
                    help="0 = whatever scores highest, 0.3 = deep cuts")
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--strict-matching", action="store_true",
                    help="refuse fuzzy title matches as model input")
    ap.add_argument("--hide-watchlist", action="store_true",
                    help="also exclude films already on their watchlist")
    args = ap.parse_args()

    profile = load_export(args.export)
    catalog = load_catalog(args.catalog)
    model = ItemItemCF.load(args.model)

    recs = recommend_for_profile(
        profile, model, TitleIndex(catalog), catalog,
        n=args.n,
        popularity_damping=args.damping,
        min_support=args.min_support,
        max_tier=Tier.YEAR_OFF if args.strict_matching else Tier.FUZZY,
        exclude_watchlist=args.hide_watchlist,
    )

    print(f"{profile}")
    print(f"{recs.matched_ratings} ratings usable, {recs.excluded} films excluded")
    if recs.used_fallback:
        print("!! too few matched ratings for CF — showing popular films instead")
    print()
    for rec in recs:
        print(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
