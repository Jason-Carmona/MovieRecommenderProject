"""From a Letterboxd export to a ranked list of films.

This is the seam between the two halves of the project: `matching` turns
titles into MovieLens ids, `cf` turns ids into scores, and this module makes
them agree about what a "film" is.

The translation chain is worth holding in your head, because a bug anywhere in
it produces recommendations that are wrong but perfectly plausible-looking:

    "Stalker" (1979)          Letterboxd export
      -> movieId 1237         matching.TitleIndex
      -> column 921           position in the trained model
      -> score 0.83           cf.ItemItemCF
      -> movieId 296          back out of the model
      -> "Pulp Fiction"       the catalog

One scale question resolves itself nicely: Letterboxd rates 0.5-5.0 in half
stars and so does MovieLens, so ratings transfer with no conversion. That is
luck, not design — check it again if you ever swap in a different corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cf import ItemItemCF
from .letterboxd import Profile
from .matching import Tier, TitleIndex, match_films
from .movielens import Movie

# Below this many matched ratings, item-item CF has too little to work from and
# its output is mostly noise dressed up as personalisation.
MIN_RATINGS_FOR_CF = 5


@dataclass
class Recommendation:
    movie: Movie
    score: float
    rank: int

    def __str__(self) -> str:
        year = f" ({self.movie.year})" if self.movie.year else ""
        return f"{self.rank:>3}. {self.movie.display}{year}  [{self.score:+.3f}]"


@dataclass
class Recommendations:
    items: list[Recommendation]
    matched_ratings: int
    excluded: int          # films of theirs kept out of the results
    used_fallback: bool    # True when there was too little history for CF

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def _positions(
    movies: list[Movie], item_pos: dict[int, int]
) -> np.ndarray:
    """MovieLens ids -> model columns, dropping films the model never saw."""
    return np.array(
        [item_pos[m.movie_id] for m in movies if m.movie_id in item_pos],
        dtype=int,
    )


def recommend_for_profile(
    profile: Profile,
    model: ItemItemCF,
    index: TitleIndex,
    catalog: list[Movie],
    n: int = 20,
    popularity_damping: float = 0.15,
    min_support: int = 2,
    max_tier: Tier = Tier.FUZZY,
    exclude_watchlist: bool = False,
    reject_ids: set[int] | None = None,
) -> Recommendations:
    """Recommend `n` films the user has not seen.

    `max_tier` controls how much matching uncertainty to accept as *input*.
    Feeding a fuzzy mismatch into the model quietly poisons the whole list —
    one wrong film drags its entire neighbourhood up the ranking — so
    tightening this to `Tier.YEAR_OFF` is the first thing to try when output
    looks strange.

    `exclude_watchlist` defaults to False, which is a deliberate choice: when a
    film the user had already added to their watchlist shows up in the results,
    that is the cheapest sanity check there is. Once you trust the model, turn
    it on, because telling someone about a film they already know about is not
    a recommendation.

    `reject_ids` are matches the user has told us are wrong. They are dropped
    from the model input *and* kept out of the results: we have just been told
    we cannot identify that film, so recommending it back would look broken even
    though the id is technically unseen.
    """
    reject_ids = reject_ids or set()
    item_pos = {int(mid): pos for pos, mid in enumerate(model.item_ids)}

    rated = match_films(profile.ratings, index)
    usable = [
        m for m in rated.matched
        if m.tier <= max_tier and m.movie.movie_id not in reject_ids
    ]

    positions = np.array(
        [item_pos[m.movie.movie_id] for m in usable if m.movie.movie_id in item_pos],
        dtype=int,
    )
    values = np.array(
        [m.film.rating for m in usable if m.movie.movie_id in item_pos],
        dtype=np.float32,
    )

    # Everything they have watched is off the table, whether or not they rated
    # it — a lot of Letterboxd users log far more than they score.
    watched = match_films(profile.watched, index)
    exclude = _positions([m.movie for m in watched.matched], item_pos)
    if reject_ids:
        rejected = np.array(
            [item_pos[mid] for mid in reject_ids if mid in item_pos], dtype=int
        )
        exclude = np.union1d(exclude, rejected) if len(rejected) else exclude
    if exclude_watchlist:
        listed = match_films(profile.watchlist, index)
        exclude = np.union1d(
            exclude, _positions([m.movie for m in listed.matched], item_pos)
        )

    by_id = {m.movie_id: m for m in catalog}
    seen_count = len(exclude)

    # The model and the catalog are built from different files and can disagree
    # about which films exist — `min_item_ratings` drops rarely-rated films from
    # training, and a catalog can be swapped for a newer one. Blocking the
    # difference up front is the only way to get a list of exactly `n` films;
    # filtering afterwards silently returns fewer, which looks like the model
    # ran out of ideas.
    nameable = np.fromiter(
        (pos for pos, mid in enumerate(model.item_ids) if int(mid) in by_id),
        dtype=int,
    )
    unnameable = np.setdiff1d(np.arange(len(model.item_ids)), nameable)
    if len(unnameable):
        exclude = np.union1d(exclude, unnameable)

    used_fallback = len(positions) < MIN_RATINGS_FOR_CF

    if used_fallback:
        # Cold start. Rather than pretend, fall back to popular films they have
        # not seen — honest, and still better than an empty page.
        blocked = set(exclude.tolist()) | set(positions.tolist())
        order = np.argsort(-model.popularity)
        picks = [int(p) for p in order if int(p) not in blocked][:n]
        scores = model.popularity[picks].astype(float)
    else:
        picked, raw_scores = model.recommend(
            positions, values, exclude=exclude, n=n,
            popularity_damping=popularity_damping, min_support=min_support,
        )
        picks = [int(p) for p in picked]
        scores = raw_scores

    out: list[Recommendation] = []
    for rank, (pos, score) in enumerate(zip(picks, scores), start=1):
        movie = by_id.get(int(model.item_ids[pos]))
        if movie is not None:
            out.append(Recommendation(movie, float(score), rank))

    return Recommendations(
        items=out,
        matched_ratings=len(positions),
        excluded=seen_count,
        used_fallback=used_fallback,
    )
