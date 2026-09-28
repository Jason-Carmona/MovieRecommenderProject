"""Load the MovieLens catalog.

MovieLens is the ratings corpus the recommender is trained on, and its
`links.csv` maps every movieId to a TMDB and IMDb id — which is what lets a
Letterboxd export reach TMDB metadata later.

Titles in `movies.csv` follow their own conventions, which `matching` has to
undo:

    Matrix, The (1999)
    Amelie (Fabuleux destin d'Amelie Poulain, Le) (2001)
    Nosferatu the Vampyre (Nosferatu: Phantom der Nacht) (1979)
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

YEAR_SUFFIX = re.compile(r"\s*\((\d{4})\)\s*$")
_PARENTHETICAL = re.compile(r"\s*\([^()]*\)")
_INVERTED_ARTICLE = re.compile(r",\s*(\w+)$")

# Articles MovieLens moves to the end of a title ("Matrix, The"). Lives here
# rather than in `matching` because the catalog is what imposes the convention;
# `matching` imports this list.
ARTICLES = {
    "the", "a", "an", "le", "la", "les", "l", "il", "lo", "gli", "der", "die",
    "das", "ein", "eine", "el", "los", "las", "un", "una", "uno", "het", "de",
    "den", "det", "os", "as",
}


def display_title(catalog_title: str) -> str:
    """Turn a MovieLens title into something you would put on a poster.

    MovieLens titles are built for sorting, not for reading:

        "Jetée, La"                              -> "La Jetée"
        "400 Blows, The (Les quatre cents coups)" -> "The 400 Blows"
        "Wild Strawberries (Smultronstället)"     -> "Wild Strawberries"

    Alternate titles in brackets are dropped and the inverted article is put
    back where a person would say it. Used only for display — matching still
    works from `raw_title`, which keeps every variant it needs.
    """
    text = _PARENTHETICAL.sub("", catalog_title).strip()
    if not text:                       # a title that was *only* a parenthetical
        text = catalog_title.strip()

    match = _INVERTED_ARTICLE.search(text)
    if match and match.group(1).casefold() in ARTICLES:
        text = f"{match.group(1)} {text[: match.start()]}".strip()
    return text


@dataclass(frozen=True)
class Movie:
    """One MovieLens film."""

    movie_id: int
    raw_title: str
    title: str
    year: int | None
    genres: tuple[str, ...] = ()
    tmdb_id: int | None = None
    imdb_id: str | None = None

    @property
    def display(self) -> str:
        """Readable title, for when TMDB has nothing better."""
        return display_title(self.title)


def _split_year(raw: str) -> tuple[str, int | None]:
    m = YEAR_SUFFIX.search(raw)
    if not m:
        return raw.strip(), None
    return raw[: m.start()].strip(), int(m.group(1))


def _int(raw: str) -> int | None:
    raw = (raw or "").strip()
    return int(raw) if raw.isdigit() else None


def load_catalog(data_dir: str | Path) -> list[Movie]:
    """Read movies.csv (+ links.csv when present) from a MovieLens directory."""
    data_dir = Path(data_dir)
    movies_csv = data_dir / "movies.csv"
    if not movies_csv.is_file():
        raise FileNotFoundError(
            f"{movies_csv} not found — run `python scripts/fetch_movielens.py`"
        )

    links: dict[int, tuple[int | None, str | None]] = {}
    links_csv = data_dir / "links.csv"
    if links_csv.is_file():
        with links_csv.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                mid = _int(row.get("movieId", ""))
                if mid is None:
                    continue
                imdb = (row.get("imdbId") or "").strip()
                links[mid] = (
                    _int(row.get("tmdbId", "")),
                    f"tt{imdb}" if imdb else None,
                )

    catalog: list[Movie] = []
    with movies_csv.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            mid = _int(row.get("movieId", ""))
            raw = (row.get("title") or "").strip()
            if mid is None or not raw:
                continue
            title, year = _split_year(raw)
            genres = tuple(
                g for g in (row.get("genres") or "").split("|")
                if g and g != "(no genres listed)"
            )
            tmdb, imdb = links.get(mid, (None, None))
            catalog.append(
                Movie(mid, raw, title, year, genres, tmdb, imdb)
            )
    return catalog
