import contextlib
import io
import numpy as np

from src.data.load import load_ratings
from src.data.split import train_test_split
from src.evaluation.metrics import rmse, precision_at_k
from src.models.matrix_factorization import MatrixFactorization


def evaluate(model, train, test, test_users):
    """Return (test RMSE, mean Precision@10) for a fitted model."""
    actual = test["rating"].values
    preds = [model.predict(r.user_id, r.movie_id) for r in test.itertuples()]
    test_rmse = rmse(actual, preds)

    p = []
    for uid in test_users:
        seen = set(train[train["user_id"] == uid]["movie_id"])
        relevant = set(test[(test["user_id"] == uid) & (test["rating"] >= 4)]["movie_id"])
        if not relevant:
            continue
        recs = model.recommend(uid, n=10, seen=seen)
        p.append(precision_at_k(recs, relevant, k=10))
    return test_rmse, np.mean(p)


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = train_test_split(ratings)
    test = test[
        test["movie_id"].isin(train["movie_id"])
        & test["user_id"].isin(train["user_id"])
    ]
    test_users = test["user_id"].unique()

    configs = [
        dict(n_factors=20, n_epochs=20, lr=0.01, reg=0.1),   # baseline
        dict(n_factors=20, n_epochs=40, lr=0.01, reg=0.1),   # more epochs
        dict(n_factors=20, n_epochs=40, lr=0.01, reg=0.05),  # less reg
        dict(n_factors=20, n_epochs=40, lr=0.01, reg=0.02),  # even less reg
        dict(n_factors=50, n_epochs=40, lr=0.01, reg=0.05),  # more factors
        dict(n_factors=50, n_epochs=60, lr=0.01, reg=0.05),  # more factors + epochs
    ]

    print(f"{'factors':>7} {'epochs':>6} {'lr':>5} {'reg':>5} {'testRMSE':>9} {'P@10':>7}")
    for cfg in configs:
        model = MatrixFactorization(seed=42, **cfg)
        # MF.fit prints per-epoch RMSE; silence it during tuning
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(train)
        test_rmse, p10 = evaluate(model, train, test, test_users)
        print(
            f"{cfg['n_factors']:>7} {cfg['n_epochs']:>6} {cfg['lr']:>5} "
            f"{cfg['reg']:>5} {test_rmse:>9.4f} {p10:>7.4f}",
            flush=True,
        )
