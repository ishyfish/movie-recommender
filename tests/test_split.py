import pandas as pd
import pytest
from src.data.split import train_test_split


@pytest.fixture
def small_ratings():
    """Minimal synthetic ratings: 4 users, varied rating counts."""
    return pd.DataFrame({
        "user_id":  [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        "movie_id": [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 1],
        "rating":   [5, 3, 4, 2, 1, 4, 5, 3, 2, 4, 3],
        "timestamp":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })


def test_no_pair_in_both_splits(small_ratings):
    """A (user_id, movie_id) pair must not appear in both train and test."""
    train, test = train_test_split(small_ratings, test_fraction=0.2, seed=42)
    train_pairs = set(zip(train["user_id"], train["movie_id"]))
    test_pairs = set(zip(test["user_id"], test["movie_id"]))
    assert train_pairs.isdisjoint(test_pairs), "Overlap found between train and test pairs"


def test_all_test_users_in_train(small_ratings):
    """Every user who appears in test must also have ratings in train."""
    train, test = train_test_split(small_ratings, test_fraction=0.2, seed=42)
    test_users = set(test["user_id"])
    train_users = set(train["user_id"])
    assert test_users.issubset(train_users), f"Test-only users: {test_users - train_users}"


def test_all_ratings_accounted_for(small_ratings):
    """Train + test should contain every row from the original (no rows dropped)."""
    train, test = train_test_split(small_ratings, test_fraction=0.2, seed=42)
    assert len(train) + len(test) == len(small_ratings)


def test_single_rating_user_stays_in_train(small_ratings):
    """User 4 has exactly one rating and must end up in train, not test."""
    train, test = train_test_split(small_ratings, test_fraction=0.2, seed=42)
    assert 4 in set(train["user_id"])
    assert 4 not in set(test["user_id"])


def test_reproducible_with_same_seed(small_ratings):
    """Same seed must produce identical splits."""
    train1, test1 = train_test_split(small_ratings, seed=0)
    train2, test2 = train_test_split(small_ratings, seed=0)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_different_seeds_differ(small_ratings):
    """Different seeds should (almost certainly) produce different splits."""
    _, test1 = train_test_split(small_ratings, seed=0)
    _, test2 = train_test_split(small_ratings, seed=99)
    # Not guaranteed but overwhelmingly likely with real data
    assert not test1.equals(test2), "Different seeds produced identical splits — suspicious"
