import pandas as pd
import numpy as np
from src.data.matrix import build_user_item_matrix
from sklearn.metrics.pairwise import cosine_similarity

class UserCF:
    def fit(self, train_df: pd.DataFrame, min_ratings: int = 20) -> None:
        """Fit the model on training ratings.

        Builds the user-item matrix and computes user-user cosine similarities.
        """ 
        # _user_idx maps user_id → row index: {1: 0, 2: 1, 3: 2, ...}
        #_item_idx maps movie_id → column index: {1: 0, 2: 1, 10: 2, ...}

        self._matrix, self._user_idx, self._item_idx = build_user_item_matrix(train_df)
        self._sim = cosine_similarity(self._matrix)
        self._global_mean = train_df["rating"].mean()
        self._idx_to_item = {idx: movie_id for movie_id, idx in self._item_idx.items()}
        self._item_counts = self._matrix.getnnz(axis=0)
        self._enough_support = self._item_counts >= min_ratings


    def predict(self, user_id: int, movie_id: int) -> float:
        u = self._user_idx[user_id]
        i = self._item_idx[movie_id]

        # get target movie column
        movie_ratings = self._matrix[:, i].toarray().flatten()
        rated = movie_ratings > 0

        sims = self._sim[u]
        numerator = (sims[rated] * movie_ratings[rated]).sum()
        denominator = sims[rated].sum()
        return numerator / denominator if denominator > 0 else self._global_mean


    def recommend(self, user_id: int, n: int, seen: set) -> list: 
        """Return top n movie IDs for user, excluding movies already seen """

        u = self._user_idx[user_id]
   
        dense_matrix = self._matrix.toarray()
        rated = (dense_matrix > 0).astype(float)
        
        numerator = self._sim[u] @ dense_matrix
        denominator = self._sim[u] @ rated

        safe_denom = np.where(denominator >0, denominator, 1)
        scores = np.where(denominator > 0, numerator / safe_denom, self._global_mean)
        scores[~self._enough_support] = -np.inf

        
        seen_indices = [self._item_idx[movie_id] for movie_id in seen if movie_id in self._item_idx]
        scores[seen_indices] = -np.inf
        top_indices = np.argsort(scores)[::-1][:n]
        recommendations = [self._idx_to_item[idx] for idx in top_indices]


        return recommendations