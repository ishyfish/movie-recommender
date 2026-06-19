import numpy as np
import pandas as pd

from src.data.load import load_ratings
from src.data.split import train_test_split
from src.evaluation.metrics import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k
from src.models.popularity import PopularityRecommender
from src.models.item_cf import ItemCF
from src.models.user_cf import UserCF

ratings = load_ratings()
train, test = train_test_split(ratings)

test = test[
    test["movie_id"].isin(train["movie_id"]) &
    test["user_id"].isin(train["user_id"])
]
test_users = test["user_id"].unique()
actual = test["rating"].values

# --- populatiry model ---

pop_model = PopularityRecommender()
pop_model.fit(train)
pop_predictions = [
    pop_model.predict(row.user_id, row.movie_id)
    for row in test.itertuples()
] 

pop_rmse = rmse(actual, pop_predictions)
pop_mae = mae(actual, pop_predictions)

pop_p, pop_r, pop_n = [], [], []

for user_id in test_users:
    seen = set(train[train["user_id"] == user_id]["movie_id"])
    relevant = set(
        test[
            (test["user_id"] == user_id) & 
            (test["rating"] >= 4)]
        ["movie_id"]
    )

    if len(relevant) == 0:
        continue

    recs = pop_model.recommend(user_id, n=10, seen = seen)
    pop_p.append(precision_at_k(recs, relevant, k=10))
    pop_r.append(recall_at_k(recs, relevant, k=10))
    pop_n.append(ndcg_at_k(recs, relevant, k=10))

# --- item-item cf model ----

item_model = ItemCF()
item_model.fit(train)
item_predictions = [
    item_model.predict(row.user_id, row.movie_id)
    for row in test.itertuples()
]

item_rmse = rmse(actual, item_predictions)
item_mae = mae(actual, item_predictions)

item_p, item_r, item_n = [], [], []

for user_id in test_users:
    seen = set(train[train["user_id"] == user_id]["movie_id"])
    relevant = set(
        test[
            (test["user_id"] == user_id) & 
            (test["rating"] >= 4)]
        ["movie_id"]
    )

    if len(relevant) == 0:
        continue

    recs = item_model.recommend(user_id, n=10, seen = seen)
    item_p.append(precision_at_k(recs, relevant, k=10))
    item_r.append(recall_at_k(recs, relevant, k=10))
    item_n.append(ndcg_at_k(recs, relevant, k=10))

# --- user-user cf model ----

user_model = UserCF()
user_model.fit(train)
user_predictions = [
    user_model.predict(row.user_id, row.movie_id) 
    for row in test.itertuples()
]

user_rmse = rmse(actual, user_predictions)
user_mae = mae(actual, user_predictions)

user_p, user_r, user_n = [], [], []

for user_id in test_users:
    seen = set(train[train["user_id"] == user_id]["movie_id"])
    relevant = set(
        test[
            (test["user_id"] == user_id) & 
            (test["rating"] >= 4)]
        ["movie_id"]
    )

    if len(relevant) == 0:
        continue

    recs = user_model.recommend(user_id, n=10, seen = seen)
    user_p.append(precision_at_k(recs, relevant, k=10))
    user_r.append(recall_at_k(recs, relevant, k=10))
    user_n.append(ndcg_at_k(recs, relevant, k=10))


print(f"{'Model':<12} {'RMSE':>6} {'MAE':>6} {'P@10':>6} {'R@10':>6} {'NDCG@10':>8}")
print(f"{'Popularity':<12} {pop_rmse:>6.4f} {pop_mae:>6.4f} {np.mean(pop_p):>6.4f} {np.mean(pop_r):>6.4f} {np.mean(pop_n):>8.4f}")
print(f"{'ItemCF':<12} {item_rmse:>6.4f} {item_mae:>6.4f} {np.mean(item_p):>6.4f} {np.mean(item_r):>6.4f} {np.mean(item_n):>8.4f}")
print(f"{'UserCF':<12} {user_rmse:>6.4f} {user_mae:>6.4f} {np.mean(user_p):>6.4f} {np.mean(user_r):>6.4f} {np.mean(user_n):>8.4f}")
