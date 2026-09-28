import numpy as np
import pytest
from scipy import sparse

from lbxd import synthetic
from lbxd.cf import ItemItemCF, center_by_user, item_similarity
from lbxd.evaluate import temporal_split


def test_centering_removes_per_user_scale():
    # Same taste, different generosity: alice rates high, bob rates low.
    matrix = sparse.csr_matrix(np.array([[5.0, 4.5, 0.0],
                                         [3.0, 2.5, 0.0]], dtype=np.float32))
    centered = center_by_user(matrix).toarray()

    np.testing.assert_allclose(centered[0, :2], centered[1, :2], atol=1e-6)
    assert centered[0, 2] == 0.0            # unrated stays unknown, not centered
    np.testing.assert_allclose(centered.sum(axis=1), [0.0, 0.0], atol=1e-6)


def test_similarity_finds_items_the_same_people_liked():
    # Items 0 and 1 are co-liked by the first three users; item 2 is separate.
    ratings = np.array([
        [5.0, 5.0, 1.0],
        [4.5, 5.0, 1.0],
        [5.0, 4.5, 0.5],
        [1.0, 1.0, 5.0],
        [0.5, 1.0, 4.5],
    ], dtype=np.float32)
    sim = item_similarity(center_by_user(sparse.csr_matrix(ratings)),
                          top_k=2, shrinkage=0.0).toarray()

    assert sim[0, 1] > 0
    assert sim[0, 0] == 0.0                 # never its own neighbour
    assert sim[0, 2] == 0.0                 # negative correlation is dropped


def test_shrinkage_penalises_thin_evidence():
    """Two items co-rated by only a few users should not score like two items
    co-rated by many, even when the pattern is identically clean.

    Three items minimum: after user-centering, a user with exactly two ratings
    always produces one positive and one negative deviation, so any two items
    are perfectly anti-correlated and the test would measure nothing.
    """
    thin = np.array([[5.0, 4.5, 1.0], [4.5, 5.0, 1.5]], dtype=np.float32)
    thick = np.repeat(thin, 30, axis=0)

    def sim_of(block, shrinkage):
        m = center_by_user(sparse.csr_matrix(block))
        return item_similarity(m, top_k=1, shrinkage=shrinkage).toarray()[0, 1]

    assert sim_of(thin, 0.0) == pytest.approx(sim_of(thick, 0.0), abs=1e-6)
    assert sim_of(thin, 25.0) < sim_of(thick, 25.0)


def test_top_k_bounds_the_model_size():
    raw = synthetic.make_ratings(n_users=80, n_items=60, seed=1)
    data = raw.to_dataset()
    model = ItemItemCF.fit(data.matrix, data.item_ids, top_k=5)

    per_item = np.diff(model.similarity.indptr)
    assert per_item.max() <= 5


@pytest.fixture(scope="module")
def model_and_data():
    raw = synthetic.make_ratings(n_users=200, n_items=150, seed=3)
    data = raw.to_dataset()
    return ItemItemCF.fit(data.matrix, data.item_ids, top_k=30), data


def test_recommendations_exclude_everything_already_seen(model_and_data):
    model, data = model_and_data
    positions, ratings = data.user_row(0)
    extra = np.array([p for p in range(data.n_items) if p not in set(positions)][:5])

    picks, _ = model.recommend(positions, ratings, exclude=extra, n=10)

    assert len(picks) == 10
    assert not set(picks) & set(positions.tolist())
    assert not set(picks) & set(extra.tolist())


def test_disliked_films_push_their_neighbours_down(model_and_data):
    model, data = model_and_data
    positions, ratings = data.user_row(0)

    # Change one opinion and hold everything else fixed. Note that we cannot
    # compare against a uniform rating vector: centering turns "rated
    # everything 5" into all zeros, which expresses no preference at all.
    adored = ratings.copy(); adored[0] = 5.0
    loved = model.score(positions, adored)

    flipped = ratings.copy(); flipped[0] = 0.5
    hated = model.score(positions, flipped)

    neighbours = model.similarity[positions[0]].indices
    assert len(neighbours) > 0
    assert hated[neighbours].mean() < loved[neighbours].mean()


def test_popularity_damping_shifts_picks_toward_the_tail(model_and_data):
    model, data = model_and_data
    positions, ratings = data.user_row(0)

    plain, _ = model.recommend(positions, ratings, n=10, popularity_damping=0.0)
    damped, _ = model.recommend(positions, ratings, n=10, popularity_damping=0.5)

    assert model.popularity[damped].mean() < model.popularity[plain].mean()


def test_min_support_suppresses_one_neighbour_flukes(model_and_data):
    model, data = model_and_data
    positions, ratings = data.user_row(0)

    loose, _ = model.recommend(positions, ratings, n=20, min_support=1)
    strict, _ = model.recommend(positions, ratings, n=20, min_support=5)

    mask = np.zeros(data.n_items, dtype=np.float32)
    mask[positions] = 1.0
    support = (model.similarity != 0).T @ mask

    # Every strict pick is backed by at least 5 of the user's own films...
    assert (support[strict] >= 5).all()
    # ...and the filter actually removed something, rather than being a no-op.
    # (Blocked items are replaced from further down, so `strict` is not a
    # subset of `loose` — it is a different list, not a shorter one.)
    assert (support[loose] < 5).any()


def test_empty_history_is_handled_not_crashed(model_and_data):
    model, _ = model_and_data
    picks, scores = model.recommend(np.array([], dtype=int), np.array([]), n=5)
    assert len(picks) == 0 and len(scores) == 0


def test_model_survives_a_round_trip(tmp_path, model_and_data):
    model, data = model_and_data
    path = tmp_path / "model.npz"
    model.save(path)
    reloaded = ItemItemCF.load(path)

    positions, ratings = data.user_row(1)
    np.testing.assert_allclose(
        model.score(positions, ratings), reloaded.score(positions, ratings), atol=1e-6
    )
    np.testing.assert_array_equal(model.item_ids, reloaded.item_ids)


def test_training_never_sees_the_holdout():
    """The guard against the bug that silently inflates every metric."""
    raw = synthetic.make_ratings(n_users=100, n_items=80, seed=5)
    split = temporal_split(raw)
    train = split.train.matrix

    for user_pos, (items, _) in split.holdout.items():
        assert train[user_pos, items].nnz == 0


def test_truncate_matches_a_fresh_fit_at_that_k(model_and_data):
    """The optimisation the parameter sweep rests on: trimming a K=30 model to
    K=5 must give exactly what fitting at K=5 would have."""
    model, data = model_and_data
    trimmed = model.truncate(5)
    refitted = ItemItemCF.fit(data.matrix, data.item_ids, top_k=5)

    np.testing.assert_allclose(
        trimmed.similarity.toarray(), refitted.similarity.toarray(), atol=1e-6
    )
    assert np.diff(trimmed.similarity.indptr).max() <= 5


def test_truncate_is_a_copy_not_a_mutation(model_and_data):
    model, _ = model_and_data
    before = model.similarity.nnz
    model.truncate(2)
    assert model.similarity.nnz == before


def test_truncating_above_the_existing_k_changes_nothing(model_and_data):
    model, _ = model_and_data
    assert model.truncate(10_000).similarity.nnz == model.similarity.nnz
