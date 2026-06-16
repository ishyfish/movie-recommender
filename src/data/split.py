import pandas as pd
from src.config import RANDOM_SEED, TEST_FRACTION


def train_test_split(
    ratings: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test using per-user holdout.

    For each user, randomly hold out `test_fraction` of their ratings as test.
    Users with only one rating are kept entirely in train — they cannot be split
    without making the model blind to them at training time.

    Guarantees:
    - Every user in the test set also appears in the train set.
    - No (user_id, movie_id) pair appears in both splits.

    Returns (train_df, test_df), both with the same columns as `ratings`.
    """
    train_rows, test_rows = [], []

    for _, user_ratings in ratings.groupby("user_id"):
        if len(user_ratings) < 2:
            # Can't hold out anything — model would be blind to this user
            train_rows.append(user_ratings)
            continue

        test_sample = user_ratings.sample(
            frac=test_fraction, random_state=seed
        )
        train_sample = user_ratings.drop(test_sample.index)
        train_rows.append(train_sample)
        test_rows.append(test_sample)

    train = pd.concat(train_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)
    return train, test
