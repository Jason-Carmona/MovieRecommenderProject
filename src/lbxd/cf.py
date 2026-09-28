"""Item-item collaborative filtering.

The idea in one sentence: if the people who liked *Stalker* also tended to like
*Solaris*, then those two films are neighbours, and someone who loved *Stalker*
should probably see *Solaris*.

Note what that does NOT use: genre, cast, director, plot, or anything else
about the films themselves. The only signal is the pattern of who rated what.
That is what makes collaborative filtering interesting — it discovers that
*Paprika* and *Perfect Blue* belong together without ever being told they share
a director, and it also discovers connections no metadata could express ("films
that appeal to people going through a specific kind of twenties").

Why item-item rather than user-user:
  * There are usually far fewer items than users, so the similarity matrix is
    smaller and can be precomputed once.
  * Item neighbourhoods are stable. Your taste shifts week to week; the fact
    that *Stalker* and *Solaris* attract the same crowd does not.
  * A brand new user can be served immediately — we look up their films in a
    matrix that was built without them. That is exactly our situation: someone
    uploads a Letterboxd export and expects recommendations now, and they are
    not in MovieLens at all.

The pipeline is four steps, each of which fixes a specific failure of the step
before it. They are documented at their implementations below:

    1. centre ratings per user      -> fixes generous vs harsh raters
    2. cosine between item columns  -> the raw similarity
    3. shrink by co-rater count     -> fixes flukes from tiny overlaps
    4. keep top-K neighbours        -> fixes noise and memory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

# Both tuned on ml-32m against 2,000 held-out users; see docs/how-it-works.md.
# K=200 beats K=100 by +4.5% NDCG@10 (paired SE 0.0013, ~4 sigma) for a 50 MB
# artifact instead of 25 MB. K=400 is a further +2% for another 50 MB, which is
# available via `truncate` but not worth it as a default.
DEFAULT_TOP_K = 200
# Shrinkage plateaus here: lambda=200 was indistinguishable from lambda=100.
DEFAULT_SHRINKAGE = 100.0
EPS = 1e-8


def center_by_user(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Subtract each user's mean rating from their own ratings.

    Step 1. Some people rate everything 4 or 5; others hand out 2s and reserve
    4 for masterpieces. Raw cosine similarity would read those two groups as
    disagreeing when they actually agree about which films are better than
    which. Letterboxd makes this worse than MovieLens does — rating culture
    there is openly personal.

    After centering, a rating means "how far above or below this person's own
    average", which is comparable across people. Cosine on centered data is
    what the literature calls *adjusted cosine similarity* (Sarwar et al.,
    2001).

    Worked example. Two users, same taste, different scales:

        raw        Stalker  Solaris        centered   Stalker  Solaris
        alice          5.0      4.5        alice         +0.25   -0.25
        bob            3.0      2.5        bob           +0.25   -0.25

    Raw, alice's vector dwarfs bob's. Centered, they are identical — which is
    the truth we want the similarity to see.

    Only stored (nonzero) entries are touched. Unrated stays unknown; centering
    an unknown into a number would invent an opinion the user never gave.
    """
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    counts = np.diff(matrix.indptr)                     # ratings per user
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    # np.repeat expands each user's mean to cover their own stored entries,
    # which lets us subtract from .data directly without a Python loop.
    matrix.data -= np.repeat(means.astype(np.float32), counts)
    return matrix


def _normalize_columns(matrix: sparse.csr_matrix) -> sparse.csc_matrix:
    """Scale each item column to unit length, so a dot product IS the cosine."""
    csc = matrix.tocsc(copy=True)
    norms = np.sqrt(np.asarray(csc.multiply(csc).sum(axis=0)).ravel())
    norms[norms < EPS] = 1.0                            # untouched items stay 0
    csc.data /= np.repeat(norms.astype(np.float32), np.diff(csc.indptr))
    return csc


def item_similarity(
    centered: sparse.csr_matrix,
    top_k: int = DEFAULT_TOP_K,
    shrinkage: float = DEFAULT_SHRINKAGE,
    block: int = 512,
) -> sparse.csr_matrix:
    """Build the sparse item-item similarity matrix. Row i holds i's neighbours.

    Step 2 — cosine. With columns normalised to unit length, the similarity of
    items i and j is just the dot product of their columns:

        sim(i, j) = sum_u c[u,i] * c[u,j]   where c is centered and normalised

    Users who rated neither contribute 0, so the sum only ever runs over people
    who rated both. That is the "do the same people feel the same way about
    these two films" question, answered in one multiplication.

    Step 3 — shrinkage. Cosine has no notion of sample size. Two obscure films
    rated by the same 3 people can score a perfect 1.0 by coincidence, and
    without correction those flukes dominate the top-K lists and the model
    recommends nonsense. So we pull every similarity toward zero in proportion
    to how little evidence supports it:

        sim'(i, j) = sim(i, j) * n_ij / (n_ij + lambda)

    where n_ij is the number of users who rated both films. With lambda = 25:
    3 co-raters keeps 11% of the score, 25 keeps 50%, 250 keeps 91%. Big,
    well-evidenced neighbourhoods survive nearly intact; coincidences collapse.
    Lambda is the single most useful knob here — raise it if recommendations
    look random, lower it if they look generic.

    Step 4 — top-K. The full matrix is n_items squared: for ml-32m that is 7.6
    billion cells, which does not fit anywhere. It is also mostly noise, since
    a film's 5000th-most-similar film tells us nothing. Keeping the K best
    neighbours per item makes the model small, fast, and *better* — truncation
    is denoising, not just compression.

    We also drop negative similarities. "People who liked A disliked B" is a
    real signal in principle, but in practice it is dominated by noise from
    sparse overlaps, and keeping it tends to push weird items up the ranking.

    Computed in blocks of columns because the intermediate is dense: one block
    against every item is `block x n_items` floats, which is the real memory
    ceiling of training.
    """
    normalized = _normalize_columns(centered)
    # Binary copy: co-rater counts are "how many users rated both", which is the
    # same dot product with every rating replaced by 1.
    binary = normalized.copy()
    binary.data = np.ones_like(binary.data)

    n_items = normalized.shape[1]
    top_k = min(top_k, n_items - 1) if n_items > 1 else 0
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        chunk = normalized[:, start:stop]

        sims = np.asarray((chunk.T @ normalized).todense(), dtype=np.float32)
        co = np.asarray((binary[:, start:stop].T @ binary).todense(), dtype=np.float32)
        sims *= co / (co + shrinkage)

        # An item is trivially its own nearest neighbour; that would recommend
        # films back to the user that they just told us they watched.
        sims[np.arange(stop - start), np.arange(start, stop)] = 0.0
        sims[sims <= 0] = 0.0

        if top_k > 0:
            # argpartition finds the K largest per row in O(n) instead of
            # sorting all n_items — the difference is large at 87k items.
            cut = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
            keep = np.zeros_like(sims, dtype=bool)
            keep[np.arange(sims.shape[0])[:, None], cut] = True
            sims[~keep] = 0.0

        r, c = np.nonzero(sims)
        rows.append(r + start)
        cols.append(c)
        vals.append(sims[r, c])

    if not rows:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)

    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, n_items),
        dtype=np.float32,
    )


@dataclass
class ItemItemCF:
    """A trained model: the similarity matrix, plus what it needs to rank.

    similarity  CSR (n_items, n_items); row i = i's K nearest neighbours
    popularity  ratings per item, used to damp blockbusters at ranking time
    item_ids    column position -> MovieLens movieId
    """

    similarity: sparse.csr_matrix
    popularity: np.ndarray
    item_ids: np.ndarray

    @classmethod
    def fit(
        cls,
        matrix: sparse.csr_matrix,
        item_ids: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        shrinkage: float = DEFAULT_SHRINKAGE,
        block: int = 512,
    ) -> "ItemItemCF":
        centered = center_by_user(matrix)
        return cls(
            similarity=item_similarity(centered, top_k, shrinkage, block),
            popularity=np.asarray((matrix != 0).sum(axis=0)).ravel(),
            item_ids=np.asarray(item_ids),
        )

    # ---- scoring ---------------------------------------------------------

    def score(
        self,
        positions: np.ndarray,
        ratings: np.ndarray,
        popularity_damping: float = 0.0,
        min_support: int = 1,
        normalize: bool = False,
    ) -> np.ndarray:
        """Score every item for one user. Returns an array of length n_items.

        The user's rated items vote for their neighbours, weighted by how much
        the user liked each one *relative to their own average*:

            score(j) = sum_{i in rated} sim(i, j) * (r_i - mean_r)

        Centering the user's ratings is what lets a bad rating push films
        *away*: give *Crash* 1.5 stars when your average is 3.5 and its weight
        is -2.0, dragging its whole neighbourhood down. An uncentered version
        could only ever add enthusiasm, never subtract it.

        `normalize` divides that sum by `sum |sim(i, j)|`, making it a weighted
        average rather than a weighted total. **It defaults to False, and the
        reason is the most important thing on this page.**

        Dividing gives a *predicted rating* — the right quantity if you are
        minimising RMSE on held-out ratings, which is what the classic
        item-based CF papers measured. It is the wrong quantity for *ranking*,
        because it discards how much evidence backs each score. A film that 30
        of your favourites point at and a film that one lucky neighbour points
        at come out identical. The second kind is far more numerous, so it
        floods the top of the list.

        Measured on ml-latest-small, this is not a subtlety:

            normalised          NDCG@10 = 0.0065   (worse than random)
            raw weighted sum    NDCG@10 = 0.1126   (+46% over popularity)

        A 17x difference from one division. The raw weighted sum is the
        standard choice for top-N recommendation (Deshpande & Karypis, 2004);
        the normalised form is kept because it is correct for rating
        prediction, and because that contrast is worth being able to reproduce.

        `min_support` guards the same failure from the other side: require at
        least that many of the user's films to neighbour an item before it may
        be recommended at all. Useful under either scoring rule.

        `popularity_damping` is documented on `recommend`.
        """
        weights = np.zeros(self.similarity.shape[0], dtype=np.float32)
        if len(positions) == 0:
            return weights

        ratings = np.asarray(ratings, dtype=np.float32)
        weights[positions] = ratings - ratings.mean()

        scores = self.similarity.T @ weights

        if normalize or min_support > 1:
            mask = np.zeros_like(weights)
            mask[positions] = 1.0
            if normalize:
                scores = scores / ((abs(self.similarity).T @ mask) + EPS)
            if min_support > 1:
                support = (self.similarity != 0).T @ mask
                scores[support < min_support] = -np.inf

        if popularity_damping > 0:
            # The penalty is scaled to this user's own score range. Raw weighted
            # sums have no fixed scale — someone with 2,000 ratings produces
            # scores an order of magnitude larger than someone with 50 — so a
            # constant subtraction would be crushing for one user and invisible
            # for another. Anchoring to the largest score present makes
            # `popularity_damping` mean the same thing for everybody: "penalise
            # the most-rated film by up to this fraction of the top score".
            finite = scores[np.isfinite(scores)]
            scale = float(np.abs(finite).max()) if finite.size else 1.0
            scores = scores - popularity_damping * scale * self._popularity_penalty()

        return scores

    def _popularity_penalty(self) -> np.ndarray:
        """Popularity on a 0-1 scale, compressed by a log.

        Raw counts are useless as a penalty: the most-rated film in MovieLens
        has ~100,000 ratings and the median has ~3, so a linear penalty would
        be one enormous cliff. Ratings counts are roughly log-normal, so log1p
        turns that into something evenly spread, and we rescale to 0-1 so the
        damping coefficient means the same thing on any dataset.
        """
        logged = np.log1p(self.popularity.astype(np.float32))
        peak = logged.max()
        return logged / peak if peak > 0 else logged

    def recommend(
        self,
        positions: np.ndarray,
        ratings: np.ndarray,
        exclude: np.ndarray | None = None,
        n: int = 20,
        popularity_damping: float = 0.0,
        min_support: int = 1,
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-n item positions and their scores, seen films removed.

        `popularity_damping` subtracts a multiple of scaled log-popularity from
        every score. At 0 the model recommends whatever scores highest, which
        in practice means *The Shawshank Redemption* to everyone — the most
        rated films have the most similarity edges and drift to the top of any
        neighbourhood ranking. Around 0.1-0.3 the list turns into things the
        user plausibly has not already heard of, which for a Letterboxd
        audience is the entire point. Push it too far and you get obscurity for
        its own sake. It is a taste knob; tune it by looking at output, then
        confirm with the eval harness that precision has not collapsed.

        `exclude` is where "films they haven't seen" is enforced: pass every
        position the user has watched, including films they never rated.

        A user with no ratings gets an empty list rather than an arbitrary one.
        Collaborative filtering has literally nothing to work from without a
        history, and every item would score an identical zero — returning the
        first few of those would be dressing up "I don't know" as an answer.
        Callers should fall back to a popularity list for cold users.
        """
        if len(positions) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        scores = self.score(
            positions, ratings, popularity_damping, min_support, normalize
        )

        blocked = np.zeros(len(scores), dtype=bool)
        blocked[positions] = True
        if exclude is not None and len(exclude):
            blocked[exclude] = True
        scores = np.where(blocked, -np.inf, scores)

        n = min(n, int((scores > -np.inf).sum()))
        if n <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]           # argpartition is unordered
        return top, scores[top]

    def truncate(self, top_k: int) -> "ItemItemCF":
        """A copy keeping only each item's `top_k` strongest neighbours.

        Truncation is monotone — the 50 best neighbours are a subset of the 200
        best — so a model fitted at K=200 already contains every smaller model.
        Sweeping K therefore costs one fit, not one per value, which is the
        difference between a 17-minute sweep and a 51-minute one.

        Also useful on its own: ship a K=50 artifact built from a K=200 fit
        without refitting, when artifact size matters more than the last
        fraction of a percent of accuracy.
        """
        sim = self.similarity.tocsr()
        rows, cols, vals = [], [], []

        for i in range(sim.shape[0]):
            start, stop = sim.indptr[i], sim.indptr[i + 1]
            data = sim.data[start:stop]
            idx = sim.indices[start:stop]
            if len(data) > top_k:
                keep = np.argpartition(-data, top_k)[:top_k]
                data, idx = data[keep], idx[keep]
            rows.append(np.full(len(idx), i))
            cols.append(idx)
            vals.append(data)

        trimmed = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=sim.shape, dtype=np.float32,
        )
        return ItemItemCF(trimmed, self.popularity, self.item_ids)

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sim = self.similarity.tocsr()
        np.savez_compressed(
            path,
            data=sim.data, indices=sim.indices, indptr=sim.indptr,
            shape=np.asarray(sim.shape),
            popularity=self.popularity, item_ids=self.item_ids,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ItemItemCF":
        with np.load(path) as z:
            sim = sparse.csr_matrix(
                (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
            )
            return cls(sim, z["popularity"], z["item_ids"])
