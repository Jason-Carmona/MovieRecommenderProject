"""Match a Letterboxd export against MovieLens and report how well it went.

    python -m lbxd.cli path/to/letterboxd-export.zip

The match rate is the number that decides whether the rest of the project is
worth building, so it gets its own command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .letterboxd import load_export
from .matching import Tier, TitleIndex, match_films
from .movielens import load_catalog

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "ml-latest-small"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export", type=Path, help="Letterboxd export zip or directory")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="MovieLens directory")
    ap.add_argument("--no-fuzzy", action="store_true", help="exact matches only")
    ap.add_argument("--show", type=int, default=15, help="rows per problem list")
    args = ap.parse_args(argv)

    profile = load_export(args.export)
    catalog = load_catalog(args.data)
    index = TitleIndex(catalog)

    print(f"{profile}")
    print(f"catalog: {len(catalog)} films, {len(index)} indexed titles\n")

    for label, films in (
        ("watched", profile.watched),
        ("rated", profile.ratings),
        ("watchlist", profile.watchlist),
    ):
        if not films:
            continue
        report = match_films(films, index, fuzzy=not args.no_fuzzy)
        print(f"{label:>9}: {report.summary()}")

        uncertain = [m for m in report.matched if m.uncertain]
        if uncertain:
            print(f"  check these {len(uncertain)} fuzzy matches:")
            for m in uncertain[: args.show]:
                print(f"    {m.film.name} ({m.film.year}) -> {m.movie.raw_title}")
        if report.unmatched:
            print(f"  no match for {len(report.unmatched)}:")
            for f in report.unmatched[: args.show]:
                print(f"    {f.name} ({f.year})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
