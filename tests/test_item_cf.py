import pandas as pd
import pytest
from src.models.item_cf import ItemCF


@pytest.fixture
def small_ratings():
    """Minimal synthetic ratings: 4 users, varied rating counts."""
    return pd.DataFrame({
        "user_id":  [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        "movie_id": [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 6],
        "rating":   [5, 3, 4, 2, 1, 4, 5, 3, 2, 4, 3],
        "timestamp":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })

# --- predict ---

def test_predict_range(small_ratings):
    model = ItemCF()
    model.fit(small_ratings)
    assert(1 <= model.predict(1,2) <= 5)

def test_predict_global_mean(small_ratings):
    model = ItemCF()
    model.fit(small_ratings)
    assert(model.predict(4,2) == model._global_mean)

# --- recommend ---

def test_recommend_count(small_ratings):
    model = ItemCF()
    model.fit(small_ratings)
    assert(len(model.recommend(1,3,{})) == 3)

def test_recommend_seen(small_ratings):
    model = ItemCF()
    model.fit(small_ratings)
    assert (1 not in model.recommend(1,2,{1}))

def test_recommend_valid_ids(small_ratings):
    model = ItemCF()
    model.fit(small_ratings)
    known_movies = set(small_ratings["movie_id"])
    assert all(movie_id in known_movies for movie_id in model.recommend(1, 2, {1}))

