from pathlib import Path

import numpy as np
import pytest

from lbxd import synthetic
from lbxd.cf import ItemItemCF
from lbxd.letterboxd import Film, Profile, load_export
from lbxd.matching import Tier, TitleIndex
from lbxd.movielens import load_catalog
from lbxd.recommend import recommend_for_profile

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(FIXTURES / "movielens")


@pytest.fixture(scope="module")
def index(catalog):
    return TitleIndex(catalog)


@pytest.fixture(scope="module")
def model(catalog):
    # A model over exactly the fixture catalog's ids (1..14).
    raw = synthetic.make_ratings(n_users=300, n_items=len(catalog), seed=13)
    data = raw.to_dataset()
    return ItemItemCF.fit(data.matrix, data.item_ids, top_k=5)


def test_never_recommends_a_film_they_have_watched(model, index, catalog):
    profile = load_export(FIXTURES / "export")
    recs = recommend_for_profile(profile, model, index, catalog, n=10)

    watched_titles = {f.name.casefold() for f in profile.watched}
    for rec in recs:
        assert rec.movie.title.casefold() not in watched_titles
    assert all(r.rank == i for i, r in enumerate(recs, start=1))


def test_thin_history_falls_back_instead_of_pretending(model, index, catalog):
    """Four ratings is not enough for CF, and saying so beats inventing a list."""
    profile = load_export(FIXTURES / "export")
    assert len(profile.ratings) == 4

    recs = recommend_for_profile(profile, model, index, catalog, n=5)
    assert recs.used_fallback
    assert len(recs) > 0                      # still returns something useful


def test_a_real_history_uses_the_model(model, index, catalog):
    profile = Profile(
        watched=[Film("The Matrix", 1999), Film("Amélie", 2001)],
        ratings=[
            Film("The Matrix", 1999, rating=5.0),
            Film("Amélie", 2001, rating=4.5),
            Film("Se7en", 1995, rating=4.0),
            Film("The Thing", 1982, rating=4.5),
            Film("Parasite", 2019, rating=5.0),
            Film("Cure", 1997, rating=3.5),
        ],
    )
    recs = recommend_for_profile(profile, model, index, catalog, n=5)

    assert not recs.used_fallback
    assert recs.matched_ratings == 6
    seen = {"The Matrix", "Amelie", "Seven", "Thing, The", "Parasite", "Cure"}
    assert not {r.movie.title for r in recs} & seen


def test_uncertain_matches_can_be_kept_out_of_the_input(model, index, catalog):
    """A fuzzy mismatch fed into the model poisons the neighbourhood it sits in,
    so it must be possible to refuse uncertain input."""
    profile = Profile(ratings=[Film("The Matri", 1999, rating=5.0)] * 6)

    loose = recommend_for_profile(profile, model, index, catalog, max_tier=Tier.FUZZY)
    strict = recommend_for_profile(profile, model, index, catalog,
                                   max_tier=Tier.YEAR_OFF)

    assert loose.matched_ratings == 6
    assert strict.matched_ratings == 0
    assert strict.used_fallback


def test_watchlist_exclusion_is_opt_in(model, index, catalog):
    profile = Profile(
        ratings=[
            Film("The Matrix", 1999, rating=5.0),
            Film("Amélie", 2001, rating=4.5),
            Film("Se7en", 1995, rating=4.0),
            Film("The Thing", 1982, rating=4.5),
            Film("Parasite", 2019, rating=5.0),
        ],
        watchlist=[Film("The Godfather", 1972)],
    )
    off = recommend_for_profile(profile, model, index, catalog, n=14)
    on = recommend_for_profile(profile, model, index, catalog, n=14,
                               exclude_watchlist=True)

    assert on.excluded > off.excluded
    assert "Godfather, The" not in {r.movie.title for r in on}


def test_films_missing_from_the_model_are_dropped_not_crashed(index, catalog):
    """The model is trained on a filtered catalog; matching is not. Ids that
    exist in movies.csv but never made it into training must be ignored."""
    raw = synthetic.make_ratings(n_users=100, n_items=6, seed=17)   # ids 1..6 only
    data = raw.to_dataset()
    small = ItemItemCF.fit(data.matrix, data.item_ids, top_k=3)

    profile = Profile(
        ratings=[
            Film("The Matrix", 1999, rating=5.0),      # id 1, in the model
            Film("Parasite", 2019, rating=4.0),        # id 13, not in the model
        ]
    )
    recs = recommend_for_profile(profile, small, index, catalog, n=5)
    assert recs.matched_ratings == 1
    assert all(r.movie.movie_id <= 6 for r in recs)


def test_scores_come_back_in_descending_order(model, index, catalog):
    profile = Profile(ratings=[
        Film("The Matrix", 1999, rating=5.0),
        Film("Amélie", 2001, rating=4.5),
        Film("Se7en", 1995, rating=2.0),
        Film("The Thing", 1982, rating=4.5),
        Film("Parasite", 2019, rating=5.0),
    ])
    scores = [r.score for r in recommend_for_profile(profile, model, index, catalog)]
    assert scores == sorted(scores, reverse=True)
    assert np.isfinite(scores).all()


def test_list_length_is_not_silently_shortened_by_catalog_gaps(index, catalog):
    """The model knows 300 ids; the fixture catalog names 14. Asking for 5 must
    return 5 nameable films, not 5 minus however many fell through."""
    raw = synthetic.make_ratings(n_users=200, n_items=300, seed=19)
    data = raw.to_dataset()
    wide = ItemItemCF.fit(data.matrix, data.item_ids, top_k=10)

    profile = Profile(ratings=[
        Film("The Matrix", 1999, rating=5.0),
        Film("Amélie", 2001, rating=4.5),
        Film("Se7en", 1995, rating=4.0),
        Film("The Thing", 1982, rating=4.5),
        Film("Parasite", 2019, rating=5.0),
        Film("Cure", 1997, rating=3.0),
    ])
    # min_support=1 because the fixture catalog is 14 films inside a 300-film
    # model: requiring two of the user's six ratings to neighbour a candidate
    # that is *also* one of those 14 is nearly impossible here. On a real
    # catalog the default of 2 is cheap and worth having.
    recs = recommend_for_profile(profile, wide, index, catalog, n=5, min_support=1)

    assert len(recs) == 5
    assert all(r.movie.movie_id in {m.movie_id for m in catalog} for r in recs)


def test_min_support_can_legitimately_empty_the_list(index, catalog):
    """Not a bug: if nothing clears the evidence bar, the honest answer is
    nothing. Worth pinning down so it is never mistaken for a silent failure."""
    raw = synthetic.make_ratings(n_users=200, n_items=300, seed=19)
    data = raw.to_dataset()
    wide = ItemItemCF.fit(data.matrix, data.item_ids, top_k=10)

    profile = Profile(ratings=[
        Film("The Matrix", 1999, rating=5.0),
        Film("Amélie", 2001, rating=4.5),
        Film("Se7en", 1995, rating=4.0),
        Film("The Thing", 1982, rating=4.5),
        Film("Parasite", 2019, rating=5.0),
        Film("Cure", 1997, rating=3.0),
    ])
    strict = recommend_for_profile(profile, wide, index, catalog, n=5, min_support=2)

    assert len(strict) == 0
    assert not strict.used_fallback          # it had history; it had no answer


def test_rejected_matches_leave_the_input_and_the_output(model, index, catalog):
    """The user's answer to the /match confirmation step. A title we got wrong
    must stop influencing the model, and must not be recommended back."""
    profile = Profile(ratings=[
        Film("The Matrix", 1999, rating=5.0),       # id 1
        Film("Amélie", 2001, rating=4.5),           # id 2
        Film("Se7en", 1995, rating=4.0),            # id 8
        Film("The Thing", 1982, rating=4.5),        # id 4
        Film("Parasite", 2019, rating=5.0),         # id 13
        Film("Cure", 1997, rating=3.0),             # id 11
    ])
    kept = recommend_for_profile(profile, model, index, catalog, n=8, min_support=1)
    dropped = recommend_for_profile(profile, model, index, catalog, n=8,
                                    min_support=1, reject_ids={2})

    assert kept.matched_ratings == 6
    assert dropped.matched_ratings == 5
    assert 2 not in {r.movie.movie_id for r in dropped}


def test_display_titles_are_presentable(catalog):
    """MovieLens titles are built for sorting; nobody wants "Jetee, La" on a
    poster. The API renders this, so it must be right at the source."""
    from lbxd.movielens import display_title

    assert display_title("Jetée, La") == "La Jetée"
    assert display_title("400 Blows, The (Les quatre cents coups)") == "The 400 Blows"
    assert display_title("Wild Strawberries (Smultronstället)") == "Wild Strawberries"
    assert display_title("M") == "M"                      # nothing to strip
    assert display_title("Am, The") == "The Am"           # article moved back

    # A title that is *only* a parenthetical must not vanish.
    assert display_title("(Untitled)") == "(Untitled)"

    # "Thing, The" -> "The Thing"; a trailing non-article stays put.
    assert display_title("Cloud Atlas, Part 2") == "Cloud Atlas, Part 2"

    by_id = {m.movie_id: m for m in catalog}
    assert by_id[1].display == "The Matrix"
    assert by_id[8].display == "Seven"
