import contextlib
import io

import numpy as np

from src.data.load import load_ratings
from src.data.split import train_test_split
from src.evaluation.metrics import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k
from src.models.popularity import PopularityRecommender
from src.models.item_cf import ItemCF
from src.models.user_cf import UserCF
from src.models.matrix_factorization import MatrixFactorization


def rating_metrics(model, test, actual):
    """RMSE and MAE of a model's predictions over the test ratings."""
    predictions = [model.predict(row.user_id, row.movie_id) for row in test.itertuples()]
    return rmse(actual, predictions), mae(actual, predictions)


def ranking_metrics(model, train, test, test_users, k=10):
    """Mean Precision@k, Recall@k, and NDCG@k over test users with relevant items.

    Relevant items are test movies the user rated >= 4. Users with none are skipped.
    """
    p, r, n = [], [], []
    for user_id in test_users:
        seen = set(train[train["user_id"] == user_id]["movie_id"])
        relevant = set(
            test[(test["user_id"] == user_id) & (test["rating"] >= 4)]["movie_id"]
        )
        if not relevant:
            continue
        recs = model.recommend(user_id, n=k, seen=seen)
        p.append(precision_at_k(recs, relevant, k=k))
        r.append(recall_at_k(recs, relevant, k=k))
        n.append(ndcg_at_k(recs, relevant, k=k))
    return np.mean(p), np.mean(r), np.mean(n)


def evaluate_model(model, train, test, test_users, actual):
    """Fit a model and return all five metrics as a tuple."""
    # MatrixFactorization prints per-epoch RMSE during fit; silence it here.
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(train)
    rmse_, mae_ = rating_metrics(model, test, actual)
    p, r, n = ranking_metrics(model, train, test, test_users)
    return rmse_, mae_, p, r, n


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = train_test_split(ratings)
    test = test[
        test["movie_id"].isin(train["movie_id"])
        & test["user_id"].isin(train["user_id"])
    ]
    test_users = test["user_id"].unique()
    actual = test["rating"].values

    models = {
        "Popularity": PopularityRecommender(),
        "ItemCF": ItemCF(),
        "UserCF": UserCF(),
        "MF": MatrixFactorization(),
    }

    print(f"{'Model':<12} {'RMSE':>6} {'MAE':>6} {'P@10':>6} {'R@10':>6} {'NDCG@10':>8}")
    for name, model in models.items():
        rmse_, mae_, p, r, n = evaluate_model(model, train, test, test_users, actual)
        print(f"{name:<12} {rmse_:>6.4f} {mae_:>6.4f} {p:>6.4f} {r:>6.4f} {n:>8.4f}")
