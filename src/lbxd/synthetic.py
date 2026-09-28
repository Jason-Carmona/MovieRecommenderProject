"""A fake MovieLens, for development and tests.

Two reasons this exists rather than being a stopgap for the download outage:

1. **Tests must not need a 250 MB download.** A test suite that depends on a
   third party's web server is a test suite that fails on someone else's laptop.

2. **Ground truth.** On real data you can measure that the model beats the
   popularity baseline, but you cannot prove *why*. Here the latent structure
   is something we chose: users belong to taste groups, films belong to
   clusters, and preference is generated from those. If collaborative filtering
   cannot rediscover a structure we planted ourselves, the implementation is
   broken — and that makes a genuinely strong test, the kind that catches a
   transposed matrix or a leaking split.

The generator deliberately includes a power-law popularity distribution, so the
popularity baseline is a real opponent here and not a straw man.
"""

from __future__ import annotations

import numpy as np

from .ratings import RawRatings


def make_ratings(
    n_users: int = 400,
    n_items: int = 300,
    n_groups: int = 5,
    ratings_per_user: int = 40,
    seed: int = 7,
) -> RawRatings:
    """Generate ratings with a planted taste structure.

    Each user belongs to one of `n_groups` taste groups and each film to one of
    `n_groups` clusters. A user rates films from their own cluster highly and
    others near the middle, with per-user bias (generous vs harsh raters, the
    thing `cf.center_by_user` exists to remove) and noise on top.
    """
    rng = np.random.default_rng(seed)

    item_cluster = rng.integers(0, n_groups, size=n_items)
    user_group = rng.integers(0, n_groups, size=n_users)

    # Power-law-ish popularity: a few films everyone has seen, a long tail
    # nobody has. Real catalogues look like this, and it is what makes the
    # popularity baseline hard to beat.
    popularity = rng.pareto(1.2, size=n_items) + 1.0
    popularity /= popularity.sum()

    users: list[int] = []
    items: list[int] = []
    values: list[float] = []
    stamps: list[int] = []

    for u in range(n_users):
        # Generous raters sit up near 4, harsh ones down near 2.5.
        bias = rng.normal(0.0, 0.5)
        n = max(10, int(rng.normal(ratings_per_user, ratings_per_user / 4)))

        # Sampling weight blends "what is popular" with "what this group
        # gravitates to", so a user's history is neither purely mainstream nor
        # purely niche.
        affinity = (item_cluster == user_group[u]).astype(float)
        weights = popularity * (1.0 + 3.0 * affinity)
        weights /= weights.sum()

        chosen = rng.choice(n_items, size=min(n, n_items), replace=False, p=weights)
        # Watch order is arbitrary, but timestamps must increase within a user
        # or the temporal split has nothing to cut on.
        base_time = int(rng.integers(1_000_000_000, 1_400_000_000))

        for step, item in enumerate(chosen):
            liked = 1.5 * affinity[item]
            score = 3.0 + liked + bias + rng.normal(0.0, 0.6)
            score = float(np.clip(np.round(score * 2) / 2, 0.5, 5.0))

            users.append(u + 1)
            items.append(int(item) + 1)
            values.append(score)
            stamps.append(base_time + step * 86_400)

    return RawRatings(
        np.asarray(users), np.asarray(items),
        np.asarray(values, dtype=np.float32), np.asarray(stamps, dtype=np.int64),
    )
