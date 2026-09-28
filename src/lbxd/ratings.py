"""The ratings matrix everything else is built on.

A recommender needs one central object: the sparse matrix R where R[u, i] is
the rating user u gave item i, and every missing entry is *unknown* — not zero,
not "disliked", unknown. That distinction is the whole difficulty of the
problem, and forgetting it is the most common way to build a broken model.

Why sparse: ml-32m is 200k users x 87k films = 17 billion cells, of which 32
million are filled (0.2%). Dense would need ~70 GB; sparse needs ~400 MB.

Why the split between `RawRatings` and `Dataset`: evaluation has to cut each
user's history in half chronologically and train on only the older part. That
cut happens on the raw event list, before any matrix exists, because building a
matrix throws away the row order that timestamps live in. `Dataset` then gets
built from a subset — always with the *same* id maps, so that column 4021 means
the same film in the training matrix as in the full one. Getting that wrong is
a silent, evaluation-invalidating bug.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class Dataset:
    """A ratings matrix plus the id <-> position translation tables.

    MovieLens ids are sparse and huge (movieId 193609 in a 9k-film catalog), so
    items and users are remapped to dense 0..n-1 positions. Everything inside
    the model works in positions; only the edges of the system speak ids.
    """

    matrix: sparse.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    def item_pos(self) -> dict[int, int]:
        """movieId -> column position."""
        return {int(mid): pos for pos, mid in enumerate(self.item_ids)}

    def popularity(self) -> np.ndarray:
        """How many users rated each item. The bias we spend effort fighting."""
        return np.asarray((self.matrix != 0).sum(axis=0)).ravel()

    def user_row(self, user_pos: int) -> tuple[np.ndarray, np.ndarray]:
        """One user's (item positions, ratings)."""
        row = self.matrix[user_pos]
        return row.indices, row.data

    def __repr__(self) -> str:
        nnz = self.matrix.nnz
        cells = self.n_users * self.n_items
        return (
            f"Dataset({self.n_users} users, {self.n_items} items, "
            f"{nnz} ratings, {nnz / cells:.3%} dense)"
        )


@dataclass
class RawRatings:
    """The rating events as a flat list, before any matrix is built."""

    users: np.ndarray
    items: np.ndarray
    values: np.ndarray
    timestamps: np.ndarray

    def __len__(self) -> int:
        return len(self.users)

    def subset(self, mask: np.ndarray) -> "RawRatings":
        return RawRatings(
            self.users[mask], self.items[mask],
            self.values[mask], self.timestamps[mask],
        )

    def ids(self) -> tuple[np.ndarray, np.ndarray]:
        """The sorted unique user and item ids present."""
        return np.unique(self.users), np.unique(self.items)

    def to_dataset(
        self,
        user_ids: np.ndarray | None = None,
        item_ids: np.ndarray | None = None,
    ) -> Dataset:
        """Build the matrix.

        Pass `user_ids`/`item_ids` to pin the index space to a wider set than
        this subset contains — which is exactly what the train split needs, so
        that a film only present in the held-out data still has a column (empty,
        and therefore never recommended, which is the honest outcome).
        """
        if user_ids is None:
            user_ids = np.unique(self.users)
        if item_ids is None:
            item_ids = np.unique(self.items)

        user_pos = np.searchsorted(user_ids, self.users)
        item_pos = np.searchsorted(item_ids, self.items)

        matrix = sparse.csr_matrix(
            (self.values.astype(np.float32), (user_pos, item_pos)),
            shape=(len(user_ids), len(item_ids)),
        )
        # A duplicated (user, item) pair would have been summed by the
        # constructor into an impossible rating like 9.0.
        matrix.sum_duplicates()
        return Dataset(matrix, user_ids, item_ids)


def _parse_csv(path: Path) -> RawRatings:
    """Parse ratings.csv. pandas' C parser, with a stdlib fallback.

    Measured on ml-32m's 877 MB / 32M rows:

        csv.DictReader    43.7s     builds a dict per row, 32 million times
        csv.reader        24.3s     same parse, no dict
        pandas.read_csv   16.5s     C parser, straight into typed arrays

    pandas is a training-time dependency only — the API serves from a trained
    .npz and never touches ratings.csv — so importing it lazily here keeps it
    off the request path entirely.
    """
    try:
        import pandas as pd

        df = pd.read_csv(
            path,
            usecols=["userId", "movieId", "rating", "timestamp"],
            dtype={"userId": np.int64, "movieId": np.int64,
                   "rating": np.float32, "timestamp": np.int64},
        )
        return RawRatings(
            df["userId"].to_numpy(), df["movieId"].to_numpy(),
            df["rating"].to_numpy(), df["timestamp"].to_numpy(),
        )
    except ImportError:
        pass

    users: list[int] = []
    items: list[int] = []
    values: list[float] = []
    stamps: list[int] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cu, ci, cr = (header.index(c) for c in ("userId", "movieId", "rating"))
        ct = header.index("timestamp") if "timestamp" in header else None
        for row in reader:
            users.append(int(row[cu]))
            items.append(int(row[ci]))
            values.append(float(row[cr]))
            stamps.append(int(float(row[ct])) if ct is not None else 0)

    return RawRatings(
        np.asarray(users), np.asarray(items),
        np.asarray(values, dtype=np.float32), np.asarray(stamps, dtype=np.int64),
    )


def load_ratings(
    data_dir: str | Path, min_item_ratings: int = 1, cache: bool = True
) -> RawRatings:
    """Read MovieLens ratings.csv, caching the parsed arrays beside it.

    Parsing 32 million rows costs ~17 seconds and produces the same four arrays
    every time. A parameter sweep re-runs this dozens of times, so the parse is
    cached as a .npz and reloaded in about a second — the difference between a
    sweep you will run and one you will not.

    Two decisions in the cache worth noting:

    * **It stores the unfiltered arrays**, and `min_item_ratings` is applied
      afterwards. Caching post-filter data would mean a cache miss every time
      you changed that number, which is exactly when you are iterating.
    * **It is invalidated by mtime**, not by a hash. Hashing 877 MB to avoid
      re-reading 877 MB saves nothing. The failure mode is a file restored with
      an older timestamp; `cache=False` is the escape hatch.
    """
    data_dir = Path(data_dir)
    path = data_dir / "ratings.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/fetch_movielens.py`"
        )

    cache_path = data_dir / ".ratings-cache.npz"
    raw: RawRatings | None = None

    if cache and cache_path.is_file():
        if cache_path.stat().st_mtime >= path.stat().st_mtime:
            with np.load(cache_path) as z:
                raw = RawRatings(
                    z["users"], z["items"], z["values"], z["timestamps"]
                )

    if raw is None:
        raw = _parse_csv(path)
        if cache:
            np.savez(
                cache_path, users=raw.users, items=raw.items,
                values=raw.values, timestamps=raw.timestamps,
            )

    if min_item_ratings > 1:
        counts = np.bincount(raw.items)
        raw = raw.subset(counts[raw.items] >= min_item_ratings)
    return raw
