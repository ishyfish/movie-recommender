import numpy as np
import pandas as pd
from src.data.matrix import build_user_item_matrix

class MatrixFactorization:
    def __init__(self, n_factors = 20, n_epochs = 30, lr = 0.01, reg = 0.1, seed = 42):
        self.n_factors = n_factors   # k — length of each latent vector
        self.n_epochs = n_epochs     # how many passes over the training data
        self.lr = lr                 # learning rate — size of each nudge
        self.reg = reg               # λ — regularization strength
        self.seed = seed             # reproducibility

    def fit(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> None:
        """Train the latent factors and biases with SGD.

        If `test_df` is given, records per-epoch train and test RMSE in
        `self.history_` so a training curve can be plotted.
        """
        self._matrix, self._user_idx, self._item_idx = build_user_item_matrix(train_df)
        self._idx_to_item = {idx: movie_id for movie_id, idx in self._item_idx.items()}

        n_users = len(self._user_idx)
        n_items = len(self._item_idx)

        self._global_mean = train_df["rating"].mean()
        self._bias_u = np.zeros(n_users)
        self._bias_i = np.zeros(n_items)

        rng = np.random.default_rng(self.seed)
        self._p = rng.normal(0, 0.1, size = (n_users, self.n_factors))
        self._q = rng.normal(0, 0.1, size = (n_items, self.n_factors))

        users = train_df["user_id"].map(self._user_idx).values
        items = train_df["movie_id"].map(self._item_idx).values
        ratings = train_df["rating"].values

        # Optional held-out set for the training curve, mapped to indices.
        # Test rows referencing unseen users/movies are dropped.
        if test_df is not None:
            t_u = test_df["user_id"].map(self._user_idx)
            t_i = test_df["movie_id"].map(self._item_idx)
            mask = t_u.notna() & t_i.notna()
            t_users = t_u[mask].astype(int).values
            t_items = t_i[mask].astype(int).values
            t_ratings = test_df["rating"][mask].values

        self.history_ = {"train": [], "test": []}

        for epoch in range(self.n_epochs):
            for u, i, r in zip(users, items, ratings):
                #1. predict
                pred = self._global_mean + self._bias_u[u] + self._bias_i[i] + self._p[u] @ self._q[i]

                #2. error
                e = r - pred

                #3. updates
                p_u = self._p[u].copy()
                self._bias_u[u] += self.lr * (e - self.reg * self._bias_u[u])
                self._bias_i[i] += self.lr * (e - self.reg * self._bias_i[i])
                self._p[u]      += self.lr * (e * self._q[i] - self.reg * self._p[u])
                self._q[i]      += self.lr * (e * p_u        - self.reg * self._q[i])

            # end of epoch: record RMSE for the training curve (vectorized)
            train_pred = (self._global_mean + self._bias_u[users] + self._bias_i[items]
                          + np.sum(self._p[users] * self._q[items], axis=1))
            self.history_["train"].append(np.sqrt(np.mean((ratings - train_pred) ** 2)))
            if test_df is not None:
                test_pred = (self._global_mean + self._bias_u[t_users] + self._bias_i[t_items]
                             + np.sum(self._p[t_users] * self._q[t_items], axis=1))
                self.history_["test"].append(np.sqrt(np.mean((t_ratings - test_pred) ** 2)))

    def predict(self, user_id: int , movie_id: int) -> float:
        u = self._user_idx.get(user_id)
        i = self._item_idx.get(movie_id)

        pred = self._global_mean
        if u is not None:
            pred += self._bias_u[u]

        if i is not None:
            pred += self._bias_i[i]

        if u is not None and i is not None:
            pred += self._p[u] @ self._q[i]

        return float(pred)
    
    def recommend (self, user_id: int, n: int, seen: set) -> list:
        u = self._user_idx.get(user_id)
        scores = self._global_mean + self._bias_u[u] + self._bias_i + self._p[u] @ self._q.T
        
        seen_indices = [self._item_idx[i] for i in seen if i in self._item_idx]
        scores[seen_indices] = -np.inf
        top_indices = np.argsort(scores)[::-1] [:n]

        return [self._idx_to_item[idx] for idx in top_indices]

    def similar_items(self, movie_id: int, n: int) -> list:
        i = self._item_idx.get(movie_id)
        if i is None:
            return []
        #cosine similarity of movie i's vector against all item vectors
        q = self._q
        target = q[i]
        sims = q @ target / (np.linalg.norm(q, axis = 1) * np.linalg.norm(target) + 1e-9)
        sims[i] = -np.inf                       #exclude movie itself
        top = np.argsort(sims)[::-1][:n]
        return [self._idx_to_item[idx] for idx in top]
    
    def fold_in (self, ratings, n_steps = 20):
        p_u = np.zeros(self.n_factors)
        b_u = 0.0

        for _ in range(n_steps):
            for movie_id, r in ratings:
                i = self._item_idx.get(movie_id)
                if i is None:
                    continue
                
                pred = self._global_mean + b_u + self._bias_i[i] + p_u @ self._q[i]
                e = r - pred

                b_u += self.lr * (e - self.reg * b_u)
                p_u += self.lr * (e * self._q[i] - self.reg * p_u)

        return p_u, b_u


    def recommend_new (self, ratings, n: int, seen: set):
        p_u, b_u = self.fold_in(ratings)
        scores = self._global_mean + b_u + self._bias_i + p_u @ self._q.T

        seen_indices = [self._item_idx[i] for i in seen if i in self._item_idx]
        scores[seen_indices] = -np.inf
        top_indices = np.argsort(scores)[::-1] [:n]

        return [self._idx_to_item[idx] for idx in top_indices]

