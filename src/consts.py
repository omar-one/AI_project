from pathlib import Path

TARGET = "salary"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

DATA_PATH_TRAIN = str(_DATA_DIR / "Train.csv")
DATA_PATH_TEST = str(_DATA_DIR / "Test.csv")

RANDOM_STATE = 42

