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

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-k recommendations that are relevant"""
    recommended = recommended[:k]
    hits = 0
    for item in recommended:
        if item in relevant:
            hits += 1
    return hits / k if len(recommended) > 0 and k > 0 else 0.0

def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in the top-K recommendations."""
    recommended = recommended[:k]
    hits = 0
    for item in recommended:
        if item in relevant:
            hits += 1
    return hits / len(relevant) if len(relevant) > 0 else 0.0

def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized discounted cumulative gain at K (binary relevance).

    Relevant items ranked higher contribute more than those ranked lower.
    Normalized so 1.0 = perfect ranking, 0.0 = no relevant items in top K.
    """
    top_k = recommended[:k]

    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i is 0-indexed; +2 so rank 1 = log2(2) = 1.0

    ideal_hits = min(len(relevant), k)
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0
