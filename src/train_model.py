"""Offline training job: fit the matrix factorization model on ALL ratings and
persist it for the API to load at startup.

Unlike evaluation (which trains on a held-out split), the served model trains on
every rating — in production there is no need to hold out a test set; every
rating should contribute to the learned factors.

Run with: python -m src.train_model
"""

from src.config import DATA_PROCESSED
from src.data.load import load_ratings
from src.models.matrix_factorization import MatrixFactorization
from src.models.persistence import save_model


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings()
    print(f"training on {len(ratings)} ratings...")

    model = MatrixFactorization()
    model.fit(ratings)

    save_model(model, str(DATA_PROCESSED))
    print(f"saved model to {DATA_PROCESSED}")
