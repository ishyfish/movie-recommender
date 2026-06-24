from pydantic import BaseModel, Field

class MovieOut(BaseModel):
    movie_id: int
    title: str

class RatingIn(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(ge = 1, le = 5)