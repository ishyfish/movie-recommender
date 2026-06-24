from fastapi import FastAPI, HTTPException
from src.models.persistence import load_model
from src.data.load import load_movies, load_ratings
from src.config import DATA_PROCESSED
from api.schemas import RatingIn

app = FastAPI()

model = load_model(str(DATA_PROCESSED))
movies = load_movies()
titles = movies.set_index("movie_id")["title"].to_dict()

ratings = load_ratings()
seen = ratings.groupby("user_id")["movie_id"].apply(set).to_dict()
submitted = []

@app.get("/recommendations/{user_id}")
def recommendations(user_id: int, n: int = 10):
    if user_id not in model._user_idx:
        raise HTTPException(status_code = 404, detail = "user not found")

    user_seen = seen.get(user_id, set())
    recs = model.recommend(user_id, n, user_seen)
    return [{"movie_id": m, "title": titles.get(m, "Unknown")} for m in recs]

@app.get("/movies/similar/{movie_id}")
def similar(movie_id: int, n: int = 10):
    if movie_id not in model._item_idx:
        raise HTTPException(status_code= 404, detail = "movie not found")
    
    sims = model.similar_items(movie_id, n)
    return [{"movie_id": m, "title": titles.get(m, "Unknown")} for m in sims]

@app.post("/ratings", status_code = 201)
def submit_ratings(rating: RatingIn):
    submitted.append(rating)
    seen.setdefault(rating.user_id, set()).add(rating.movie_id)
    return {"status": "accepted"}
