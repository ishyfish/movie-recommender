from surprise import Dataset, Reader, SVD
from src.data.load import load_ratings
from src.data.split import train_test_split
from src.evaluation.metrics import rmse, mae

ratings = load_ratings()
train, test = train_test_split(ratings)

reader = Reader(rating_scale=(1,5))
training_data = Dataset.load_from_df(train[["user_id", "movie_id", "rating"]], reader)
trainset = training_data.build_full_trainset()

model = SVD(n_factors=20, n_epochs = 30, lr_all = 0.01, reg_all = 0.1, random_state = 42)
model.fit(trainset)

preds = [model.predict(row.user_id, row.movie_id).est for row in test.itertuples()]
actual = test["rating"].values

default_model = SVD(random_state=42)
default_model.fit(trainset)
default_preds = [default_model.predict(r.user_id, r.movie_id).est for r in test.itertuples()]

print(f"surprise SVD  RMSE={rmse(actual, preds):.4f}  MAE={mae(actual, preds):.4f}")
print(f"surprise SVD (defaults)  RMSE={rmse(actual, default_preds):.4f}  MAE={mae(actual, default_preds):.4f}")

