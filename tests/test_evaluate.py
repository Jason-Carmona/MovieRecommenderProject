import numpy as np
import pytest

from lbxd import synthetic
from lbxd.cf import ItemItemCF
from lbxd.evaluate import (
    PopularityRecommender, RandomRecommender, compare, evaluate, ndcg_at_k,
    precision_at_k, recall_at_k, temporal_split,
)


def test_precision_and_recall_answer_different_questions():
    recs = np.array([1, 2, 3, 4, 5])
    relevant = {2, 5, 9}

    assert precision_at_k(recs, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(recs, relevant, 5) == pytest.approx(2 / 3)
    assert precision_at_k(recs, set(), 5) == 0.0
    assert recall_at_k(recs, set(), 5) == 0.0


def test_ndcg_rewards_hits_near_the_top():
    early = ndcg_at_k(np.array([1, 9, 9]), {1}, 3)
    late = ndcg_at_k(np.array([9, 9, 1]), {1}, 3)

    assert early == pytest.approx(1.0)      # the only hit, in the best slot
    assert late == pytest.approx(0.5)       # 1/log2(4) over 1/log2(2)
    assert early > late
    assert ndcg_at_k(np.array([9, 9, 9]), {1}, 3) == 0.0


def test_ndcg_is_capped_by_what_was_achievable():
    """A user with one relevant film can still score 1.0 — the ideal ranking is
    one hit at the top, not ten."""
    assert ndcg_at_k(np.array([7, 1, 2]), {7}, 3) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def split():
    return temporal_split(synthetic.make_ratings(n_users=250, n_items=180, seed=11))


def test_split_holds_out_the_most_recent_ratings():
    raw = synthetic.make_ratings(n_users=60, n_items=50, seed=2)
    split = temporal_split(raw, holdout_frac=0.2, min_ratings=10)

    user_ids, item_ids = raw.ids()
    for user_pos, (items, _) in split.holdout.items():
        uid = user_ids[user_pos]
        mine = raw.users == uid
        held = set(item_ids[items].tolist())

        latest_train = max(
            t for t, i in zip(raw.timestamps[mine], raw.items[mine])
            if i not in held
        )
        earliest_held = min(
            t for t, i in zip(raw.timestamps[mine], raw.items[mine])
            if i in held
        )
        assert earliest_held > latest_train


def test_short_histories_are_not_evaluated():
    """Splitting eight ratings into six and two measures variance, not skill."""
    raw = synthetic.make_ratings(n_users=40, n_items=40, seed=4)
    split = temporal_split(raw, min_ratings=10_000)
    assert split.holdout == {}
    assert split.train.matrix.nnz == len(raw)


def test_train_and_holdout_share_one_index_space():
    raw = synthetic.make_ratings(n_users=50, n_items=45, seed=6)
    split = temporal_split(raw)
    _, item_ids = raw.ids()

    assert split.train.n_items == len(item_ids)
    np.testing.assert_array_equal(split.train.item_ids, item_ids)
    for items, _ in split.holdout.values():
        assert items.max() < split.train.n_items


def test_cf_beats_popularity_which_beats_random(split):
    """The headline claim, on data whose structure we planted ourselves."""
    model = ItemItemCF.fit(split.train.matrix, split.train.item_ids, top_k=50)

    cf = evaluate(model, split, "cf")
    popular = evaluate(PopularityRecommender(split.train), split, "popularity")
    random = evaluate(RandomRecommender(split.train.n_items), split, "random")

    assert cf.ndcg > popular.ndcg > random.ndcg
    assert cf.recall > popular.recall
    assert cf.users > 0


def test_popularity_baseline_recommends_almost_nothing(split):
    """Coverage is why accuracy alone is not enough: the baseline can look
    respectable while only ever naming the same few dozen films."""
    popular = evaluate(PopularityRecommender(split.train), split, "popularity")
    cf = evaluate(ItemItemCF.fit(split.train.matrix, split.train.item_ids), split, "cf")

    # Compare the two, don't hard-code a threshold: coverage is a fraction of
    # the catalog, and this synthetic catalog has 180 films where ml-32m has
    # 87,000. The same behaviour that scores 0.17 here scores ~0.001 there.
    assert popular.coverage < cf.coverage / 3
    assert cf.coverage > 0.5
    assert popular.novelty > cf.novelty      # baseline lives on the head


def test_damping_trades_accuracy_for_obscurity(split):
    model = ItemItemCF.fit(split.train.matrix, split.train.item_ids, top_k=50)

    plain = evaluate(model, split, "plain", popularity_damping=0.0)
    damped = evaluate(model, split, "damped", popularity_damping=0.5)

    assert damped.novelty < plain.novelty
    assert damped.ndcg <= plain.ndcg


def test_metrics_are_zero_when_nothing_can_be_scored(split):
    class Useless:
        def recommend(self, positions, ratings, exclude=None, n=20):
            return np.array([], dtype=int), np.array([])

    result = evaluate(Useless(), split, "useless")
    assert (result.precision, result.recall, result.ndcg) == (0.0, 0.0, 0.0)


def test_user_sampling_is_seeded_and_bounded():
    """Sampling decides who is scored, never who trains the model."""
    raw = synthetic.make_ratings(n_users=200, n_items=120, seed=8)
    full = temporal_split(raw)
    sampled = temporal_split(raw, max_users=25)
    again = temporal_split(raw, max_users=25)

    assert len(full.holdout) > 25
    assert len(sampled.holdout) == 25
    assert set(sampled.holdout) <= set(full.holdout)
    assert set(sampled.holdout) == set(again.holdout)      # reproducible
    assert sampled.train.matrix.nnz == full.train.matrix.nnz


def test_stderr_shrinks_with_more_users(split):
    """SE = std/sqrt(n). The metric's precision is bounded by how many users
    were scored, which is what makes a small sweep difference unreadable."""
    model = ItemItemCF.fit(split.train.matrix, split.train.item_ids, top_k=50)
    result = evaluate(model, split, "cf")

    assert result.ndcg_stderr > 0
    assert len(result.ndcg_values) == result.users
    expected = result.ndcg_values.std(ddof=1) / np.sqrt(len(result.ndcg_values))
    assert result.ndcg_stderr == pytest.approx(expected)


def test_a_model_compared_with_itself_shows_exactly_no_difference(split):
    model = ItemItemCF.fit(split.train.matrix, split.train.item_ids, top_k=50)
    a = evaluate(model, split, "a")
    b = evaluate(model, split, "b")

    mean, stderr, significant = compare(a, b)
    assert mean == 0.0 and stderr == 0.0 and significant is False


def test_pairing_is_tighter_for_similar_configs(split):
    """The reason `compare` exists, and the condition under which it helps.

    Pairing removes variance the two systems *share*: var(d) = var(a) + var(b)
    - 2cov(a, b). The saving comes entirely from the covariance, so it is large
    for two near-identical configs (the sweep case, where the same users are
    easy or hard for both) and vanishes for two unrelated models — where paired
    SE can be *larger* than either mean's own. This asserts the case the sweep
    actually relies on.
    """
    full = ItemItemCF.fit(split.train.matrix, split.train.item_ids, top_k=50)
    a = evaluate(full, split, "k50")
    b = evaluate(full.truncate(45), split, "k45")

    _, paired_se, _ = compare(a, b)

    assert paired_se < a.ndcg_stderr / 2
    assert paired_se < b.ndcg_stderr / 2


def test_pairing_does_not_help_for_unrelated_models(split):
    """The flip side, pinned down so nobody assumes pairing is always tighter."""
    full = ItemItemCF.fit(split.train.matrix, split.train.item_ids, top_k=50)
    a = evaluate(full, split, "strong")
    b = evaluate(full.truncate(1), split, "crippled")

    _, paired_se, _ = compare(a, b)
    assert paired_se >= min(a.ndcg_stderr, b.ndcg_stderr)


def test_comparing_different_user_sets_is_refused(split):
    """Silently misaligned pairing would invent significance out of nothing."""
    model = ItemItemCF.fit(split.train.matrix, split.train.item_ids, top_k=50)
    a = evaluate(model, split, "a")

    other = temporal_split(synthetic.make_ratings(n_users=120, n_items=90, seed=21))
    b = evaluate(
        ItemItemCF.fit(other.train.matrix, other.train.item_ids, top_k=50), other, "b"
    )

    with pytest.raises(ValueError, match="same users"):
        compare(a, b)
