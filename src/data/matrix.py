import pandas as pd
from scipy.sparse import csr_matrix

def build_user_item_matrix(
    ratings: pd.DataFrame
) -> tuple[csr_matrix, dict, dict]:
    """Builds a matrix with user ids as rows, movie ids as columns, and cells being ratings values.
        
    Returns the sparse matrix, a dict mapping user_id to row index, and a dict mapping movie_id to column index.
    """
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(ratings["user_id"].unique()))}
    item_to_idx = {movie_id: idx for idx, movie_id in enumerate(sorted(ratings["movie_id"].unique()))}

    row = ratings["user_id"].map(user_to_idx)
    col = ratings["movie_id"].map(item_to_idx)
    data = ratings["rating"]

    matrix = csr_matrix((data, (row, col)), shape = (len(user_to_idx), len(item_to_idx)))
    return matrix, user_to_idx, item_to_idx