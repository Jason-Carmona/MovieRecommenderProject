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

# The widest gap between two release years still worth calling the same film.
# Andrei Rublev spans 1966-1971; past that it is a remake or a namesake.
MAX_YEAR_DRIFT = 5


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


def _query_forms(name: str) -> list[str]:
    """Normalised forms to look a Letterboxd title up under, best first.

    Letterboxd brackets alternate titles too — "Birdman or (The Unexpected
    Virtue of Ignorance)" — so the bracketed parts and what remains outside
    them get looked up as well as the whole name.
    """
    forms: list[str] = []
    for part in [name, _PARENS.sub("", name), *_PARENS.findall(name)]:
        norm = normalize(part)
        if norm and norm not in forms:
            forms.append(norm)
    return forms


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
            # `title`, not `raw_title`: the raw form ends in "(1917)", which
            # title_variants would read as an alternate title and index every
            # film under its own release year.
            for variant in title_variants(movie.title):
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
        forms = _query_forms(film.name)
        if not forms:
            return None

        far: Movie | None = None
        for norm in forms:
            candidates = self._by_title.get(norm, [])
            if not candidates:
                continue
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
            # list, so the model cheerfully recommends it back to them. Keep it
            # as a fallback, marked uncertain so the UI can ask.
            #
            # Past MAX_YEAR_DRIFT it is a remake or a namesake, not a disputed
            # date: "Nosferatu (2024)" is not the 1922 film, and calling it one
            # feeds the model a rating for a film the user never saw.
            dated = [
                m for m in candidates
                if m.year is not None and abs(m.year - film.year) <= MAX_YEAR_DRIFT
            ]
            if dated and far is None:
                far = min(dated, key=lambda m: abs(m.year - film.year))

        # A near-miss title from the right year beats the exact title from the
        # wrong one: Letterboxd's "Oldboy (2003)" is MovieLens's "Old Boy (2003)",
        # not the 2013 remake that happens to share its spelling.
        if fuzzy:
            hit = self._fuzzy(forms[0], film.year)
            if hit is not None:
                return hit, Tier.FUZZY
        if far is not None:
            return far, Tier.YEAR_FAR
        return None

    def _fuzzy(self, norm: str, year: int | None) -> Movie | None:
        # Restrict fuzzy search to plausible release years so we compare against
        # hundreds of titles rather than ninety thousand.
        pool = self._year_window(year, span=1)
        close = difflib.get_close_matches(norm, pool, n=1, cutoff=FUZZY_CUTOFF)
        if not close:
            return None
        for movie in self._by_title[close[0]]:
            if year is None or movie.year is None or abs(movie.year - year) <= 1:
                return movie
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
