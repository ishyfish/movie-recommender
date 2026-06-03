from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

MOVIELENS_RATINGS = DATA_RAW / "ml-100k" / "u.data"
MOVIELENS_MOVIES = DATA_RAW / "ml-100k" / "u.item"

RANDOM_SEED = 42
TEST_FRACTION = 0.2
