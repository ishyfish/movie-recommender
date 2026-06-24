from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_recommendations_known_user():
    r = client.get("/recommendations/1?n=5")
    assert r.status_code == 200
    assert len(r.json()) == 5

def test_recommendations_unknown_user():
    r = client.get("/recommendations/999999?n=5")
    assert r.status_code == 404

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