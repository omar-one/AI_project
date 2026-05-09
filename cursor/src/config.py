"""Shared constants for paths, columns, and model defaults."""

from pathlib import Path

# Repo layout: AI_project/data/Test.csv and AI_project/cursor/src/config.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TARGET = "salary"
DATA_PATH = str(_PROJECT_ROOT / "data" / "Test.csv")

CAT_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
NUM_COLS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

SALARY_MAP = {"<=50K": 0, ">50K": 1}

DT_PARAM_GRID = {
    "max_depth":              [10, 15, 20],
    "min_samples_split":      [2, 5, 10],
    "min_samples_leaf":       [1, 2, 4],
}
