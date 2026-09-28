import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbxd.letterboxd import Film, load_export  # noqa: E402
from lbxd.matching import Tier, TitleIndex, match_films, normalize, title_variants  # noqa: E402
from lbxd.movielens import load_catalog  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def index():
    return TitleIndex(load_catalog(FIXTURES / "movielens"))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("The Matrix", "matrix"),
        ("Matrix, The", "matrix the"),    # comma form is handled by variants
        ("Amélie", "amelie"),
        ("WALL·E", "wall e"),
        ("WALL-E", "wall e"),
        ("Fast & Furious", "fast and furious"),
        ("Le Samouraï", "samourai"),
        ("...", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_title_variants_indexes_alternates_and_moved_articles():
    variants = title_variants("Amelie (Fabuleux destin d'Amelie Poulain, Le)")
    assert "amelie" in variants
    assert "fabuleux destin d amelie poulain" in variants

    assert "matrix" in title_variants("Matrix, The")
    assert "girl with the dragon tattoo" in title_variants(
        "Girl with the Dragon Tattoo, The (Man som hatar kvinnor)"
    )


def test_catalog_parses_year_genres_and_links():
    catalog = {m.movie_id: m for m in load_catalog(FIXTURES / "movielens")}
    matrix = catalog[1]
    assert (matrix.title, matrix.year) == ("Matrix, The", 1999)
    assert matrix.tmdb_id == 603 and matrix.imdb_id == "tt0133093"
    assert "Sci-Fi" in matrix.genres
    assert catalog[12].year is None            # no year in title
    assert catalog[10].tmdb_id is None         # blank tmdbId column


@pytest.mark.parametrize(
    "name, year, movie_id, tier",
    [
        ("The Matrix", 1999, 1, Tier.EXACT),
        ("Amélie", 2001, 2, Tier.EXACT),
        ("Nosferatu the Vampyre", 1979, 3, Tier.EXACT),
        ("WALL-E", 2008, 6, Tier.EXACT),
        ("The Girl with the Dragon Tattoo", 2009, 7, Tier.EXACT),
        ("Parasite", 2019, 13, Tier.EXACT),
        ("Blade Runner 2049", 2018, 14, Tier.YEAR_OFF),   # festival vs release
        ("Cure", 1997, 11, Tier.EXACT),
        ("Kyua", 1997, 11, Tier.EXACT),                   # alternate title
        ("Se7en", 1995, 8, Tier.EXACT),                   # a.k.a. alternate
        ("Seven", 1995, 8, Tier.EXACT),
        ("The Matri", 1999, 1, Tier.FUZZY),               # typo
    ],
)
def test_lookup_tiers(index, name, year, movie_id, tier):
    hit = index.lookup(Film(name, year))
    assert hit is not None, f"{name} ({year}) did not match"
    assert (hit[0].movie_id, hit[1]) == (movie_id, tier)


def test_year_disambiguates_remakes(index):
    assert index.lookup(Film("The Thing", 1982))[0].movie_id == 4
    assert index.lookup(Film("The Thing", 2011))[0].movie_id == 5


def test_undated_letterboxd_entry_falls_back_to_title_only(index):
    movie, tier = index.lookup(Film("The Godfather", None))
    assert movie.movie_id == 9 and tier == Tier.TITLE_ONLY


def test_unknown_title_is_not_forced_into_a_match(index):
    assert index.lookup(Film("A Film That Does Not Exist", 2023)) is None


def test_fuzzy_respects_the_year_window(index):
    # Right-ish title, wildly wrong year: better to report nothing.
    assert index.lookup(Film("The Matri", 1930)) is None


def test_fuzzy_can_be_disabled(index):
    assert index.lookup(Film("The Matri", 1999), fuzzy=False) is None


def test_report_over_a_real_export(index):
    profile = load_export(FIXTURES / "export")
    report = match_films(profile.watched, index)

    assert len(report.unmatched) == 1
    assert report.unmatched[0].name == "A Film That Does Not Exist"
    assert report.rate == pytest.approx(11 / 12)
    assert report.by_tier()[Tier.FUZZY] == 1
    assert "matched 11/12" in report.summary()


def test_a_contested_release_year_still_matches(index):
    """Andrei Rublev: finished 1966, Cannes 1969, Soviet release 1971. Refusing
    the match is not the safe choice — an unmatched film the user has seen never
    reaches the exclude list, so it gets recommended back to them."""
    movie, tier = index.lookup(Film("Blade Runner 2049", 2022))
    assert movie.movie_id == 14 and tier == Tier.YEAR_FAR


def test_a_far_year_match_is_flagged_for_confirmation(index):
    report = match_films([Film("Blade Runner 2049", 2022)], index)
    assert report.matched[0].uncertain is True
    # A one-year gap is routine and should not pester the user.
    assert match_films([Film("Blade Runner 2049", 2018)], index).matched[0].uncertain is False


def test_an_exact_year_still_wins_over_a_far_one(index):
    """The remake guard must survive the new fallback."""
    assert index.lookup(Film("The Thing", 1982))[0].movie_id == 4
    assert index.lookup(Film("The Thing", 2011))[0].movie_id == 5
