import pandas as pd

class PopularityRecommender:

    def fit(self, train: pd.DataFrame) -> None:
        self._movie_mean = train.groupby("movie_id")["rating"].mean()
        self._popularity = train.groupby("movie_id")["rating"].count().sort_values(ascending=False)
        self._global_mean = train["rating"].mean()


    def predict(self, user_id: int, movie_id: int) -> float:
        return self._movie_mean.get(movie_id, self._global_mean)
    
    def recommend(self, user_id: int, n: int, seen: set) -> list:
        results = []
        for movie_id in self._popularity.index:
            if movie_id not in seen:
                results.append(movie_id)
            if len(results) == n:
                break
        return results
