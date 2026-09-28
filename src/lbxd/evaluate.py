"""Offline evaluation: is this model actually better than doing nothing?

Recommendations are dangerously easy to fool yourself about. The output is a
list of plausible film titles, and plausible film titles look good whether the
model learned anything or not — which is why so many recommender projects ship
with no evidence at all. This module exists to make the question falsifiable.

The measuring stick is a **popularity baseline**: ignore the user entirely and
recommend the most-rated films. It is trivial, it has no personalisation, and
it is surprisingly hard to beat, because popular films are popular for the
reason that most people like them. A personalised model that cannot beat it is
not doing its job. Reporting only your model's precision, with nothing to
compare it to, tells the reader nothing.

The protocol:

  1. Split each user's history *by time*, holding out their most recent 20%.
  2. Train on the older ratings only.
  3. Ask for 10 recommendations, excluding everything in their training half.
  4. Check how many of the held-out films they actually liked turn up.

Step 1 is the one people get wrong. A random split leaks the future into the
past: the model gets to see that you loved *Aftersun* in 2023 while predicting
what you watched in 2019. A temporal split asks the real question — given
everything you had seen by some date, what did you go on to watch next?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .ratings import Dataset, RawRatings

LIKED = 4.0  # a held-out film counts as a hit only if the user rated it this high


class Recommender(Protocol):
    """What the harness needs from anything it evaluates."""

    def recommend(
        self, positions: np.ndarray, ratings: np.ndarray,
        exclude: np.ndarray | None = ..., n: int = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...


# ---- the split ----------------------------------------------------------


@dataclass
class Split:
    """Training data plus each evaluated user's held-out future."""

    train: Dataset
    holdout: dict[int, tuple[np.ndarray, np.ndarray]]  # user_pos -> (items, ratings)

    def __repr__(self) -> str:
        return (
            f"Split(train={self.train.matrix.nnz} ratings, "
            f"{len(self.holdout)} evaluated users)"
        )


def temporal_split(
    raw: RawRatings,
    holdout_frac: float = 0.2,
    min_ratings: int = 20,
    max_users: int | None = None,
    seed: int = 0,
) -> Split:
    """Hold out each active user's most recent `holdout_frac` of ratings.

    Users with fewer than `min_ratings` are left entirely in the training set:
    they stay useful as evidence for item similarities, but we do not try to
    score them, because splitting eight ratings into six and two measures
    nothing but variance.

    `max_users` evaluates a random sample of held-out users instead of all of
    them. Scoring one user is a sparse matrix-vector product over the whole
    catalog, so ml-32m's 200,000 users would take hours while telling you
    almost nothing more than 2,000 do — the metrics are means, and their
    standard error shrinks with the square root of the sample. Sampling is
    seeded so runs stay comparable.

    Note that sampling only reduces who is *scored*. Every rating still trains
    the model, including those of users we never evaluate.

    The index space is pinned to the *full* dataset's ids so that positions
    mean the same thing in both halves.
    """
    user_ids, item_ids = raw.ids()
    order = np.lexsort((raw.timestamps, raw.users))    # by user, then by time

    is_test = np.zeros(len(raw), dtype=bool)
    boundaries = np.flatnonzero(np.diff(raw.users[order])) + 1
    for chunk in np.split(order, boundaries):
        if len(chunk) < min_ratings:
            continue
        n_test = max(1, int(round(len(chunk) * holdout_frac)))
        is_test[chunk[-n_test:]] = True                 # the most recent ones

    train = raw.subset(~is_test).to_dataset(user_ids, item_ids)
    test = raw.subset(is_test)

    test_user_pos = np.searchsorted(user_ids, test.users)
    test_item_pos = np.searchsorted(item_ids, test.items)

    evaluated = np.unique(test_user_pos)
    if max_users is not None and len(evaluated) > max_users:
        evaluated = np.random.default_rng(seed).choice(
            evaluated, size=max_users, replace=False
        )

    holdout: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for u in evaluated:
        sel = test_user_pos == u
        holdout[int(u)] = (test_item_pos[sel], test.values[sel])

    return Split(train, holdout)


# ---- metrics ------------------------------------------------------------


def precision_at_k(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    """Of the k films we recommended, what fraction were ones they liked?

    Answers "how much of this list is worth the user's time". Note the ceiling:
    if a user only has 3 relevant held-out films, precision@10 can never exceed
    0.3, so the absolute number always looks low. Compare it to the baseline,
    never to 1.0.
    """
    if k == 0:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / k


def recall_at_k(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    """Of the films they went on to like, what fraction did we surface?"""
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def ndcg_at_k(recommended: np.ndarray, relevant: set[int], k: int) -> float:
    """Precision that cares about *where* in the list the hits landed.

    Discounted Cumulative Gain sums a reward for each hit, discounted by how
    far down the list it appeared:

        DCG@k = sum_{i=1..k} hit_i / log2(i + 1)

    A hit at position 1 is worth 1.0, at position 2 it is worth 0.63, at
    position 10, 0.29. Normalising by the best achievable arrangement (all hits
    at the top) gives NDCG, which lands in 0-1 and is comparable across users
    with different numbers of relevant films.

    This is the metric to lead with for a ranked list, because it is the only
    one of the three that notices the difference between a great first
    recommendation and a great tenth one — and users only ever look at the top.
    """
    if not relevant:
        return 0.0
    gains = np.array([1.0 if item in relevant else 0.0 for item in recommended[:k]])
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())

    ideal_hits = min(k, len(relevant))
    idcg = float((1.0 / np.log2(np.arange(2, ideal_hits + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


# ---- baselines ----------------------------------------------------------


class PopularityRecommender:
    """Recommend the most-rated films, identically, to everyone.

    The bar any personalised model has to clear.
    """

    def __init__(self, train: Dataset) -> None:
        self.ranking = np.argsort(-train.popularity())

    def recommend(self, positions, ratings, exclude=None, n=20):
        blocked = set(np.asarray(positions).tolist())
        if exclude is not None:
            blocked |= set(np.asarray(exclude).tolist())
        picks = [i for i in self.ranking if i not in blocked][:n]
        return np.asarray(picks, dtype=int), np.arange(len(picks), 0, -1)


class RandomRecommender:
    """The floor. If a model cannot beat this, something is wired backwards."""

    def __init__(self, n_items: int, seed: int = 0) -> None:
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def recommend(self, positions, ratings, exclude=None, n=20):
        blocked = set(np.asarray(positions).tolist())
        if exclude is not None:
            blocked |= set(np.asarray(exclude).tolist())
        allowed = np.setdiff1d(np.arange(self.n_items), np.fromiter(blocked, int))
        picks = self.rng.choice(allowed, size=min(n, len(allowed)), replace=False)
        return picks, np.zeros(len(picks), dtype=np.float32)


# ---- the harness --------------------------------------------------------


@dataclass
class Result:
    """Averages over every evaluated user."""

    name: str
    precision: float
    recall: float
    ndcg: float
    coverage: float     # share of the catalog that ever gets recommended
    novelty: float      # mean scaled log-popularity of recommendations, 0-1
    users: int
    k: int
    # Per-user NDCG, in evaluation order, so two Results over the same Split can
    # be compared *paired*. Without this a sweep cannot tell a real improvement
    # from sampling noise, and will confidently report whichever config got the
    # luckier users.
    ndcg_values: np.ndarray = field(default_factory=lambda: np.array([]))
    user_order: tuple[int, ...] = ()

    @property
    def ndcg_stderr(self) -> float:
        """Standard error of the mean NDCG.

        The metric is an average over users, so its precision is bounded by how
        many users were scored: SE = std / sqrt(n). Any two configs within a
        couple of standard errors of each other are indistinguishable, however
        confidently the table sorts them.
        """
        if len(self.ndcg_values) < 2:
            return 0.0
        return float(self.ndcg_values.std(ddof=1) / np.sqrt(len(self.ndcg_values)))

    def __str__(self) -> str:
        return (
            f"{self.name:<24} P@{self.k}={self.precision:.4f}  "
            f"R@{self.k}={self.recall:.4f}  NDCG@{self.k}={self.ndcg:.4f}  "
            f"cov={self.coverage:.3f}  nov={self.novelty:.3f}"
        )


def compare(a: Result, b: Result) -> tuple[float, float, bool]:
    """Paired comparison of two Results over the same users.

    Returns (mean difference b - a, standard error of that difference,
    significant at roughly 95%).

    Pairing matters more than it looks. Users differ enormously in how
    predictable they are, and that between-user variance swamps the difference
    between two configs. Comparing the two *means* drowns in it; comparing each
    user against themselves cancels it out.

    The saving comes from covariance, not from pairing as such:

        var(b - a) = var(a) + var(b) - 2 cov(a, b)

    Two near-identical configs are strongly correlated — the same users are
    easy or hard for both — so the covariance term is large and the paired
    standard error collapses. On ml-32m, comparing lambda=100/K=200 against the
    old default moved the SE from 0.0039 to 0.0013, turning an unreadable
    result into a 4-sigma one. For two *unrelated* models the covariance is
    near zero and pairing buys nothing, so do not read a paired SE as
    automatically the tighter number.
    """
    if a.user_order != b.user_order:
        raise ValueError("results were not evaluated over the same users")
    if len(a.ndcg_values) < 2:
        return 0.0, 0.0, False

    diff = b.ndcg_values - a.ndcg_values
    mean = float(diff.mean())
    stderr = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    return mean, stderr, bool(stderr > 0 and abs(mean) > 1.96 * stderr)


def evaluate(
    model: Recommender, split: Split, name: str, k: int = 10, **recommend_kwargs
) -> Result:
    """Score one recommender over every held-out user.

    Two diagnostics ride along with the accuracy metrics, because accuracy
    alone will happily reward a degenerate model:

    coverage — the fraction of the catalog that appears in *anyone's* list. The
    popularity baseline scores about 0.001: it recommends the same handful of
    films forever. A recommender that only knows about 200 films is not much of
    a recommender, however good its precision.

    novelty — mean scaled log-popularity of what got recommended, so lower is
    more obscure. This is the number that shows popularity damping working:
    turn the knob up and watch novelty fall, then check what it cost in NDCG.
    """
    popularity_scale = np.log1p(split.train.popularity().astype(np.float64))
    peak = popularity_scale.max() or 1.0

    order: list[int] = []
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    novelties: list[float] = []
    recommended_items: set[int] = set()
    scored_users = 0

    for user_pos, (held_items, held_ratings) in split.holdout.items():
        relevant = {int(i) for i, r in zip(held_items, held_ratings) if r >= LIKED}
        if not relevant:
            # They watched things but liked none of them. There is no right
            # answer to score against, so this user is not evidence either way.
            continue

        positions, ratings = split.train.user_row(user_pos)
        if len(positions) == 0:
            continue

        picks, _ = model.recommend(positions, ratings, exclude=positions, n=k,
                                   **recommend_kwargs)
        picks = np.asarray(picks, dtype=int)

        precisions.append(precision_at_k(picks, relevant, k))
        recalls.append(recall_at_k(picks, relevant, k))
        ndcgs.append(ndcg_at_k(picks, relevant, k))
        if len(picks):
            novelties.append(float(popularity_scale[picks].mean() / peak))
        recommended_items.update(picks.tolist())
        order.append(int(user_pos))
        scored_users += 1

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    return Result(
        name=name,
        precision=mean(precisions),
        recall=mean(recalls),
        ndcg=mean(ndcgs),
        coverage=len(recommended_items) / split.train.n_items,
        novelty=mean(novelties),
        users=scored_users,
        k=k,
        ndcg_values=np.asarray(ndcgs, dtype=np.float64),
        user_order=tuple(order),
    )
