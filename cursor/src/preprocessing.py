"""Feature and target preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from .config import CAT_COLS, NUM_COLS, SALARY_MAP


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train = X_train.replace("?", np.nan)
    X_test = X_test.replace("?", np.nan)

    imputer_cat = SimpleImputer(strategy="most_frequent")
    imputer_num = SimpleImputer(strategy="median")

    X_train[CAT_COLS] = imputer_cat.fit_transform(X_train[CAT_COLS])
    X_test[CAT_COLS] = imputer_cat.transform(X_test[CAT_COLS])

    X_train[NUM_COLS] = imputer_num.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = imputer_num.transform(X_test[NUM_COLS])

    for col in CAT_COLS:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    y_train_enc = y_train.astype(str).str.strip().map(SALARY_MAP).astype(int)
    y_test_enc = y_test.astype(str).str.strip().map(SALARY_MAP).astype(int)

    return X_train, X_test, y_train_enc, y_test_enc
