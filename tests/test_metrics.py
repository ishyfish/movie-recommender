import math
import pytest
from src.evaluation.metrics import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k


# --- RMSE ---

def test_rmse_perfect():
    assert rmse([3, 4, 5], [3, 4, 5]) == pytest.approx(0.0)

def test_rmse_known():
    # errors [0, 2], squares [0, 4], mean 2.0, sqrt ~1.414
    assert rmse([3, 5], [3, 3]) == pytest.approx(math.sqrt(2))

def test_rmse_penalizes_large_errors():
    # one big error scores worse than two small errors of same total
    assert rmse([0], [2]) > rmse([0, 0], [1, 1])


# --- MAE ---

def test_mae_perfect():
    assert mae([3, 4, 5], [3, 4, 5]) == pytest.approx(0.0)

def test_mae_known():
    # absolute errors [0, 2], mean 1.0
    assert mae([3, 5], [3, 3]) == pytest.approx(1.0)


# --- Precision@K ---

def test_precision_at_k_all_hits():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

def test_precision_at_k_no_hits():
    assert precision_at_k([1, 2, 3], {4, 5, 6}, k=3) == pytest.approx(0.0)

def test_precision_at_k_partial():
    # top 3 are [1,2,3], hits are 1 and 3 → 2/3
    assert precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=3) == pytest.approx(2 / 3)

def test_precision_at_k_empty():
    assert precision_at_k([], {1, 2, 3}, k=3) == pytest.approx(0.0)


# --- Recall@K ---

def test_recall_at_k_all_hits():
    assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

def test_recall_at_k_no_hits():
    assert recall_at_k([1, 2, 3], {4, 5, 6}, k=3) == pytest.approx(0.0)

def test_recall_at_k_partial():
    # 2 hits out of 3 relevant → 2/3
    assert recall_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=3) == pytest.approx(2 / 3)

def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)


# --- NDCG@K ---

def test_ndcg_at_k_perfect():
    # all relevant items ranked first → 1.0
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

def test_ndcg_at_k_no_hits():
    assert ndcg_at_k([1, 2, 3], {4, 5, 6}, k=3) == pytest.approx(0.0)

def test_ndcg_at_k_rank_matters():
    # hit at rank 1 should score higher than hit at rank 3
    score_rank1 = ndcg_at_k([1, 2, 3], {1}, k=3)
    score_rank3 = ndcg_at_k([2, 3, 1], {1}, k=3)
    assert score_rank1 > score_rank3
