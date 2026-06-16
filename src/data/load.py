import pandas as pd
from src.config import MOVIELENS_RATINGS, MOVIELENS_MOVIES


def load_ratings() -> pd.DataFrame:
    """Load raw ratings from u.data.

    Returns a DataFrame with columns: user_id, movie_id, rating, timestamp.
    Ratings are integers 1-5; timestamp is Unix seconds.
    No filtering or splitting — caller decides what to do with the data.
    """
    return pd.read_csv(
        MOVIELENS_RATINGS,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )


def load_movies() -> pd.DataFrame:
    """Load movie metadata from u.item.

    Returns a DataFrame with columns: movie_id, title, release_date, and
    19 binary genre columns. Encoding is latin-1 (required by this file).
    """
    genre_cols = [
        "unknown", "action", "adventure", "animation", "childrens",
        "comedy", "crime", "documentary", "drama", "fantasy",
        "film_noir", "horror", "musical", "mystery", "romance",
        "sci_fi", "thriller", "war", "western",
    ]
    cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols
    df = pd.read_csv(
        MOVIELENS_MOVIES,
        sep="|",
        names=cols,
        encoding="latin-1",
        usecols=["movie_id", "title", "release_date"] + genre_cols,
    )
    return df
