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


