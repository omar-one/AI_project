"""Shared constants for paths, columns, and model defaults."""

TARGET = "salary"
DATA_PATH = "../data/Test.csv"

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
