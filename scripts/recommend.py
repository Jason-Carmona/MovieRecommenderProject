#!/usr/bin/env python3
"""Recommend films for one Letterboxd export.

    python scripts/recommend.py ~/Downloads/letterboxd-export.zip

Expects a model from scripts/train.py and a catalog from the same MovieLens
directory it was trained on.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbxd.api import Settings                                     # noqa: E402
from lbxd.cf import ItemItemCF                                    # noqa: E402
from lbxd.letterboxd import load_export                           # noqa: E402
from lbxd.matching import Tier, TitleIndex                        # noqa: E402
from lbxd.movielens import load_catalog                           # noqa: E402
from lbxd.recommend import (                                      # noqa: E402
    FEATURE_HEADROOM, features_only, recommend_for_profile,
)
from lbxd.tmdb import MetadataCache, TMDBClient                   # noqa: E402

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
    # There is nobody here to confirm a "did you mean...?", so uncertain matches
    # stay out of the model input unless asked for. They still keep the film out
    # of the results either way.
    ap.add_argument("--use-uncertain", action="store_true",
                    help="also feed fuzzy and far-year title matches to the model")
    ap.add_argument("--hide-watchlist", action="store_true",
                    help="also exclude films already on their watchlist")
    ap.add_argument("--shorts-and-tv", action="store_true",
                    help="also recommend TV series and short films")
    args = ap.parse_args()

    profile = load_export(args.export)
    catalog = load_catalog(args.catalog)
    model = ItemItemCF.load(args.model)

    # Same .env and cache as the API: telling a series or a short from a
    # feature takes TMDB's answer.
    settings = Settings.from_env()
    tmdb = TMDBClient(settings.tmdb_api_key, settings.tmdb_read_token,
                      MetadataCache(settings.cache_path))
    screen = tmdb.enabled and not args.shorts_and_tv

    recs = recommend_for_profile(
        profile, model, TitleIndex(catalog), catalog,
        n=args.n * FEATURE_HEADROOM if screen else args.n,
        popularity_damping=args.damping,
        min_support=args.min_support,
        max_tier=Tier.FUZZY if args.use_uncertain else Tier.TITLE_ONLY,
        exclude_watchlist=args.hide_watchlist,
    )

    if screen:
        metadata = asyncio.run(
            tmdb.fetch_many([r.movie.tmdb_id for r in recs if r.movie.tmdb_id])
        )
        recs = features_only(recs, metadata, args.n)

    print(f"{profile}")
    if not tmdb.enabled and not args.shorts_and_tv:
        print("!! no TMDB key, so TV series and shorts are not filtered out")
    print(f"{recs.matched_ratings} ratings usable, {recs.excluded} films excluded")
    if recs.used_fallback:
        print("!! too few matched ratings for CF — showing popular films instead")
    print()
    for rec in recs:
        print(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
