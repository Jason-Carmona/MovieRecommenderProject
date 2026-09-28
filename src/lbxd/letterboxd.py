"""Parse a Letterboxd data export.

Letterboxd gives every user a zip from Settings -> Import & Export. We read it
directly (zip or unpacked directory) so nothing here depends on their API,
which has been invite-only for years.

Files we care about, all with a `Name,Year,Letterboxd URI` shape:
    watched.csv    every film marked watched
    ratings.csv    + `Rating` (0.5 - 5.0, half stars)
    diary.csv      + `Rating`, `Rewatch`, `Watched Date`
    watchlist.csv  films the user wants to see

Anything else in the export (reviews, likes, comments, profile) is ignored.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

WANTED = ("watched", "ratings", "diary", "watchlist")


@dataclass(frozen=True)
class Film:
    """One film as Letterboxd names it, before any ID matching."""

    name: str
    year: int | None
    uri: str = ""
    rating: float | None = None

    @property
    def key(self) -> tuple[str, int | None]:
        return (self.name.strip().casefold(), self.year)


@dataclass
class Profile:
    """What one user has seen, rated, and wants to see."""

    watched: list[Film] = field(default_factory=list)
    ratings: list[Film] = field(default_factory=list)
    watchlist: list[Film] = field(default_factory=list)

    @property
    def seen_keys(self) -> set[tuple[str, int | None]]:
        """Everything to exclude from recommendations.

        A rated film is watched by definition, but `ratings.csv` and
        `watched.csv` drift for some accounts, so union both.
        """
        return {f.key for f in self.watched} | {f.key for f in self.ratings}

    def __repr__(self) -> str:
        return (
            f"Profile(watched={len(self.watched)}, rated={len(self.ratings)}, "
            f"watchlist={len(self.watchlist)})"
        )


def _year(raw: str) -> int | None:
    raw = (raw or "").strip()
    return int(raw) if raw.isdigit() else None


def _rating(raw: str) -> float | None:
    raw = (raw or "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _rows(text: str) -> list[Film]:
    films: list[Film] = []
    for row in csv.DictReader(io.StringIO(text)):
        name = (row.get("Name") or "").strip()
        if not name:
            continue
        films.append(
            Film(
                name=name,
                year=_year(row.get("Year", "")),
                uri=(row.get("Letterboxd URI") or "").strip(),
                rating=_rating(row.get("Rating", "")),
            )
        )
    return films


def _sources(path: Path) -> dict[str, str]:
    """Map stem -> csv text for the files we want, from a zip or a directory."""
    found: dict[str, str] = {}
    if path.is_dir():
        for stem in WANTED:
            f = path / f"{stem}.csv"
            if f.is_file():
                found[stem] = f.read_text(encoding="utf-8-sig")
        return found

    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            stem = Path(info.filename).stem.lower()
            # Exports nest files one directory deep about half the time.
            if stem in WANTED and stem not in found:
                found[stem] = z.read(info).decode("utf-8-sig")
    return found


def load_export(path: str | Path) -> Profile:
    """Read a Letterboxd export zip or unpacked directory into a Profile."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no Letterboxd export at {path}")

    found = _sources(path)
    if not found:
        raise ValueError(
            f"{path} has none of {', '.join(f'{s}.csv' for s in WANTED)} — "
            "is it a Letterboxd export?"
        )

    profile = Profile(
        watched=_rows(found.get("watched", "")),
        ratings=_rows(found.get("ratings", "")),
        watchlist=_rows(found.get("watchlist", "")),
    )

    # diary.csv carries ratings for films some accounts never wrote to
    # ratings.csv. Fold in anything new, newest entry wins.
    if "diary" in found:
        have = {f.key for f in profile.ratings}
        for film in _rows(found["diary"]):
            if film.rating is not None and film.key not in have:
                profile.ratings.append(film)
                have.add(film.key)

    return profile
