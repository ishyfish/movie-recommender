import numpy as np

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error between actual and predicted ratings."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual-predicted)**2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error between actual and predicted ratings."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual-predicted)))