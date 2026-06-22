import matplotlib.pyplot as plt

from src.config import ROOT
from src.data.load import load_ratings
from src.data.split import train_test_split
from src.models.matrix_factorization import MatrixFactorization


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = train_test_split(ratings)

    # Train long enough to see test RMSE bottom out and turn back up.
    model = MatrixFactorization(n_factors=20, n_epochs=60, lr=0.01, reg=0.1)
    model.fit(train, test_df=test)

    epochs = range(1, model.n_epochs + 1)
    plt.plot(epochs, model.history_["train"], label="train RMSE")
    plt.plot(epochs, model.history_["test"], label="test RMSE")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title("Matrix Factorization training curve")
    plt.legend()

    out_dir = ROOT / "plots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "mf_training_curve.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")

    best_epoch = min(range(model.n_epochs), key=lambda e: model.history_["test"][e]) + 1
    print(f"lowest test RMSE at epoch {best_epoch}: {min(model.history_['test']):.4f}")
