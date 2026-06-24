from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_recommendations_known_user():
    r = client.get("/recommendations/1?n=5")
    assert r.status_code == 200
    body = r.json()
    assert body["personalized"] is True
    assert len(body["movies"]) == 5

def test_recommendations_unknown_user():
    r = client.get("/recommendations/999999?n=5")
    assert r.status_code == 200
    body = r.json()
    assert body["personalized"] is False
    assert len(body["movies"]) == 5

def test_similar_known_movie():
    r = client.get("/movies/similar/1?n=5")
    assert r.status_code == 200
    assert len(r.json()) == 5

def test_similar_unknown_movie():
    r = client.get("/movies/similar/999999?n=5")
    assert r.status_code == 404

def test_submit_ratings_valid():
    r = client.post("/ratings", json = {"user_id": 1, "movie_id": 700, "rating": 5})
    assert r.status_code == 201

def test_submit_ratings_invalid():
    r = client.post("/ratings", json = {"user_id": 1, "movie_id": 700, "rating": 7})
    assert r.status_code == 422

def test_movies_default_limit():
    r = client.get("/movies")
    assert r.status_code == 200
    assert len(r.json()) == 50

def test_movies_custom_limit():
    r = client.get("/movies?limit=5")
    assert r.status_code == 200
    assert len(r.json()) == 5

def test_movies_pagination_no_overlap():
    page1 = client.get("/movies?skip=0&limit=5").json()
    page2 = client.get("/movies?skip=5&limit=5").json()
    ids1 = {m["movie_id"] for m in page1}
    ids2 = {m["movie_id"] for m in page2}
    assert ids1.isdisjoint(ids2)

def test_movies_search():
    r = client.get("/movies?q=star wars")
    assert r.status_code == 200
    titles = [m["title"] for m in r.json()]
    assert all("star wars" in t.lower() for t in titles)
    assert "Star Wars (1977)" in titles

def test_movies_search_no_match():
    r = client.get("/movies?q=zzzznotamovie")
    assert r.status_code == 200
    assert r.json() == []

def test_movies_skip_out_of_range():
    r = client.get("/movies?skip=99999")
    assert r.status_code == 200
    assert r.json() == []

def test_movie_out_shape():
    r = client.get("/movies?limit=1")
    movie = r.json()[0]
    assert set(movie.keys()) == {"movie_id", "title"}
    assert isinstance(movie["movie_id"], int)
    assert isinstance(movie["title"], str)
