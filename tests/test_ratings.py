import os
import time

import numpy as np
import pytest

from lbxd.ratings import load_ratings

CSV = """userId,movieId,rating,timestamp
1,10,4.0,1000
1,20,3.5,1001
2,10,5.0,1002
2,30,2.0,1003
3,10,4.5,1004
"""


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "ratings.csv").write_text(CSV)
    return tmp_path


def test_reads_every_column(data_dir):
    raw = load_ratings(data_dir)
    assert len(raw) == 5
    np.testing.assert_array_equal(raw.users, [1, 1, 2, 2, 3])
    np.testing.assert_array_equal(raw.items, [10, 20, 10, 30, 10])
    np.testing.assert_allclose(raw.values, [4.0, 3.5, 5.0, 2.0, 4.5])
    np.testing.assert_array_equal(raw.timestamps, [1000, 1001, 1002, 1003, 1004])


def test_cache_is_written_then_reused(data_dir):
    cache = data_dir / ".ratings-cache.npz"
    assert not cache.exists()

    load_ratings(data_dir)
    assert cache.exists()

    # Corrupt the source; a reused cache means we never look at it again.
    (data_dir / "ratings.csv").write_text("garbage")
    os.utime(data_dir / "ratings.csv", (0, 0))          # keep the cache newer
    assert len(load_ratings(data_dir)) == 5


def test_cache_holds_unfiltered_data(data_dir):
    """Changing min_item_ratings must not invalidate the cache — that is
    precisely the parameter you iterate on."""
    load_ratings(data_dir, min_item_ratings=1)
    mtime = (data_dir / ".ratings-cache.npz").stat().st_mtime

    filtered = load_ratings(data_dir, min_item_ratings=3)
    assert (data_dir / ".ratings-cache.npz").stat().st_mtime == mtime
    assert set(filtered.items.tolist()) == {10}         # only film 10 has 3

    assert len(load_ratings(data_dir, min_item_ratings=1)) == 5


def test_a_newer_csv_invalidates_the_cache(data_dir):
    load_ratings(data_dir)

    time.sleep(0.01)
    (data_dir / "ratings.csv").write_text(CSV + "4,40,1.0,1005\n")
    assert len(load_ratings(data_dir)) == 6


def test_cache_can_be_bypassed(data_dir):
    load_ratings(data_dir)
    (data_dir / "ratings.csv").write_text(CSV + "4,40,1.0,1005\n")
    os.utime(data_dir / "ratings.csv", (0, 0))          # cache looks fresh

    assert len(load_ratings(data_dir)) == 5             # trusts the stale cache
    assert len(load_ratings(data_dir, cache=False)) == 6


def test_missing_file_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_movielens"):
        load_ratings(tmp_path)


def test_stdlib_fallback_matches_pandas(data_dir, monkeypatch):
    """The fallback path is not dead code — it must produce identical arrays."""
    import builtins

    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("pandas disabled for this test")
        return real_import(name, *args, **kwargs)

    with_pandas = load_ratings(data_dir, cache=False)
    monkeypatch.setattr(builtins, "__import__", no_pandas)
    without = load_ratings(data_dir, cache=False)

    np.testing.assert_array_equal(with_pandas.users, without.users)
    np.testing.assert_array_equal(with_pandas.items, without.items)
    np.testing.assert_allclose(with_pandas.values, without.values)
    np.testing.assert_array_equal(with_pandas.timestamps, without.timestamps)
