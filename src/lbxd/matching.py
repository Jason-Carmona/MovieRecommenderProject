"""Match Letterboxd titles to MovieLens ids.

There is no shared identifier between a Letterboxd export and MovieLens, so
this is fuzzy title work, and it is where the match rate is won or lost.
Normalisation has to absorb:

    trailing articles     "Matrix, The"  ->  "matrix"
    alternate titles      "Amelie (Fabuleux destin ..., Le)"  -> both indexed
    aka markers           "Seven (a.k.a. Se7en)"  ->  found as "Se7en"
    diacritics            "Amelie" == "Amélie"
    punctuation           "WALL-E" == "WALL E",  "&" == "and"
    year drift            festival vs release vs re-release, sometimes years apart

Matches are tiered so callers can trade recall for precision: EXACT is safe to
trust blindly, FUZZY is worth surfacing to the user.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum

from .letterboxd import Film
from .movielens import ARTICLES, Movie

_TRAILING_ARTICLE = re.compile(r",\s*(\w+)\s*$")
# MovieLens flags alternate titles as "Seven (a.k.a. Se7en)".
_AKA_PREFIX = re.compile(r"^a\.?\s*k\.?\s*a\.?\s*", re.IGNORECASE)
_PARENS = re.compile(r"\(([^()]*)\)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Below this ratio a difflib suggestion is noise, not a title.
FUZZY_CUTOFF = 0.88


class Tier(IntEnum):
    """How much to trust a match. Lower is better."""

    EXACT = 0       # same normalised title, same year
    YEAR_OFF = 1    # same title, year within +/- 1
    TITLE_ONLY = 2  # same title, no year to check against
    YEAR_FAR = 3    # same title, year off by more than 1
    FUZZY = 4       # near-miss title within the year window


def _strip_article(text: str) -> str:
    head, _, rest = text.partition(" ")
    return rest if rest and head in ARTICLES else text


def normalize(title: str) -> str:
    """Fold a title to its comparable form. Empty if nothing survives."""
    text = unicodedata.normalize("NFKD", title)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold().replace("&", " and ")
    text = _NON_ALNUM.sub(" ", text).strip()
    return _strip_article(text)


def title_variants(raw_title: str) -> list[str]:
    """Every normalised form a MovieLens title should be findable under.

    "Amelie (Fabuleux destin d'Amelie Poulain, Le)" indexes as both the primary
    title and the parenthesised alternate, each with its article restored and
    any "a.k.a." marker dropped.
    """
    alternates = _PARENS.findall(raw_title)
    primary = _PARENS.sub("", raw_title)

    variants: list[str] = []
    for candidate in [primary, *alternates]:
        candidate = _AKA_PREFIX.sub("", candidate.strip()).strip()
        if not candidate:
            continue
        # MovieLens parks the article after a comma: "Matrix, The".
        moved = _TRAILING_ARTICLE.sub("", candidate).strip()
        for form in {candidate, moved}:
            norm = normalize(form)
            if norm and norm not in variants:
                variants.append(norm)
    return variants


@dataclass
class Match:
    film: Film
    movie: Movie
    tier: Tier

    @property
    def uncertain(self) -> bool:
        """Worth showing the user before it influences anything."""
        return self.tier >= Tier.YEAR_FAR


@dataclass
class MatchReport:
    matched: list[Match] = field(default_factory=list)
    unmatched: list[Film] = field(default_factory=list)

    @property
    def rate(self) -> float:
        total = len(self.matched) + len(self.unmatched)
        return len(self.matched) / total if total else 0.0

    def by_tier(self) -> dict[Tier, int]:
        counts = {t: 0 for t in Tier}
        for m in self.matched:
            counts[m.tier] += 1
        return counts

    def summary(self) -> str:
        counts = self.by_tier()
        total = len(self.matched) + len(self.unmatched)
        parts = ", ".join(f"{t.name.lower()} {counts[t]}" for t in Tier)
        return (
            f"matched {len(self.matched)}/{total} ({self.rate:.1%}) — {parts}, "
            f"unmatched {len(self.unmatched)}"
        )


class TitleIndex:
    """Lookup structure over a MovieLens catalog, built once and reused."""

    def __init__(self, catalog: list[Movie]) -> None:
        self._by_title: dict[str, list[Movie]] = defaultdict(list)
        self._by_year: dict[int | None, set[str]] = defaultdict(set)

        for movie in catalog:
            for variant in title_variants(movie.raw_title):
                self._by_title[variant].append(movie)
                self._by_year[movie.year].add(variant)

    def __len__(self) -> int:
        return len(self._by_title)

    def _year_window(self, year: int | None, span: int) -> set[str]:
        """Normalised titles released within +/- span of year."""
        if year is None:
            return set().union(*self._by_year.values()) if self._by_year else set()
        window: set[str] = set()
        for offset in range(-span, span + 1):
            window |= self._by_year.get(year + offset, set())
        return window

    def lookup(self, film: Film, fuzzy: bool = True) -> tuple[Movie, Tier] | None:
        """Best match for one Letterboxd film, or None."""
        norm = normalize(film.name)
        if not norm:
            return None

        candidates = self._by_title.get(norm, [])
        if candidates:
            if film.year is None:
                return candidates[0], Tier.TITLE_ONLY
            for span, tier in ((0, Tier.EXACT), (1, Tier.YEAR_OFF)):
                hits = [
                    m for m in candidates
                    if m.year is not None and abs(m.year - film.year) <= span
                ]
                if hits:
                    return hits[0], tier
            # Title is right but every candidate year is far off.
            undated = [m for m in candidates if m.year is None]
            if undated:
                return undated[0], Tier.TITLE_ONLY

            # Films with a contested release year are common enough to matter:
            # Andrei Rublev was finished in 1966, shown at Cannes in 1969 and
            # released in the USSR in 1971, and Letterboxd and MovieLens picked
            # different ones. Refusing the match is not the safe option — an
            # unmatched film the user has *seen* never makes it onto the exclude
            # list, so the model cheerfully recommends it back to them. Match it,
            # but mark it uncertain so the UI can ask.
            dated = [m for m in candidates if m.year is not None]
            if dated:
                closest = min(dated, key=lambda m: abs(m.year - film.year))
                return closest, Tier.YEAR_FAR

        if not fuzzy:
            return None

        # Restrict fuzzy search to plausible release years so we compare against
        # hundreds of titles rather than ninety thousand.
        pool = self._year_window(film.year, span=1)
        close = difflib.get_close_matches(norm, pool, n=1, cutoff=FUZZY_CUTOFF)
        if not close:
            return None
        for movie in self._by_title[close[0]]:
            if film.year is None or movie.year is None:
                return movie, Tier.FUZZY
            if abs(movie.year - film.year) <= 1:
                return movie, Tier.FUZZY
        return None


def match_films(
    films: list[Film], index: TitleIndex, fuzzy: bool = True
) -> MatchReport:
    """Resolve a list of Letterboxd films against the catalog."""
    report = MatchReport()
    for film in films:
        hit = index.lookup(film, fuzzy=fuzzy)
        if hit is None:
            report.unmatched.append(film)
        else:
            movie, tier = hit
            report.matched.append(Match(film, movie, tier))
    return report
