from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from src.models.persistence import load_model
from src.models.popularity import PopularityRecommender
from src.data.load import load_movies, load_ratings
from src.config import DATA_PROCESSED
from api.schemas import MovieOut, RatingIn, RecommendationsOut

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins = ["http://localhost:5173"], 
    allow_methods = ["*"], 
    allow_headers = ["*"]
)

ratings = load_ratings()
seen = ratings.groupby("user_id")["movie_id"].apply(set).to_dict()
submitted = []

model = load_model(str(DATA_PROCESSED))
movies = load_movies()
titles = movies.set_index("movie_id")["title"].to_dict()

pop = PopularityRecommender()
pop.fit(ratings)


@app.get("/recommendations/{user_id}", response_model = RecommendationsOut)
def recommendations(user_id: int, n: int = 10):
    user_seen = seen.get(user_id, set())

    if user_id in model._user_idx:
        recs = model.recommend(user_id, n, user_seen)
        personalized = True
    else:
        recs = pop.recommend(user_id, n, user_seen)
        personalized = False

    return {"personalized": personalized, "movies": [{"movie_id": m, "title": titles.get(m, "Unknown")} for m in recs]}

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

@app.get("/movies", response_model = list[MovieOut])
def list_movies(skip: int = 0, limit: int = 50, q: str | None = None):
    items = list(titles.items())

    if q is not None:
        items = [(movie_id, title) for movie_id, title in items if q.lower() in title.lower()]

    items = items[skip: skip + limit]
    return [{"movie_id": movie_id, "title": title} for movie_id, title in items]