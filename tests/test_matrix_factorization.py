import numpy as np
import pandas as pd
import pytest

from src.models.matrix_factorization import MatrixFactorization


@pytest.fixture
def small_ratings():
    """Minimal synthetic ratings: 4 users, varied rating counts."""
    return pd.DataFrame({
        "user_id":  [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        "movie_id": [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 6],
        "rating":   [5, 3, 4, 2, 1, 4, 5, 3, 2, 4, 3],
        "timestamp":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })


# --- fit ---

def test_fit_records_history(small_ratings):
    model = MatrixFactorization(n_epochs=20)
    model.fit(small_ratings, test_df=small_ratings)
    assert len(model.history_["train"]) == 20
    assert len(model.history_["test"]) == 20

def test_fit_reduces_error(small_ratings):
    # SGD should drive training RMSE down over epochs
    model = MatrixFactorization(n_epochs=20)
    model.fit(small_ratings)
    assert model.history_["train"][-1] < model.history_["train"][0]


# --- predict ---

def test_predict_returns_finite(small_ratings):
    model = MatrixFactorization()
    model.fit(small_ratings)
    assert np.isfinite(model.predict(1, 2))

def test_predict_cold_start_is_global_mean(small_ratings):
    # Unknown user AND unknown movie → no biases/vectors → global mean
    model = MatrixFactorization()
    model.fit(small_ratings)
    assert model.predict(999, 999) == pytest.approx(model._global_mean)


# --- recommend ---

def test_recommend_count(small_ratings):
    model = MatrixFactorization()
    model.fit(small_ratings)
    assert len(model.recommend(1, 3, set())) == 3

def test_recommend_excludes_seen(small_ratings):
    model = MatrixFactorization()
    model.fit(small_ratings)
    assert 1 not in model.recommend(1, 2, {1})
