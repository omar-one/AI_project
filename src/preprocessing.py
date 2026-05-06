from __future__ import annotations

import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from consts import TARGET


# splitting function
def split_train_test(
    df: pd.DataFrame,
    *,
    target: str = TARGET
):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


#phase 1 cleaning
def preprocessing_phase_1(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()
    semantic = {
        "work-class": "workclass",
        "position": "occupation",
        "gender": "sex",
        "country": "native-country",
        # final sample weight naming variants (often mis-labeled vs UCI Adult "fnlwgt")
        "fnl-wgt": "fnlwgt",
        "final-weight": "fnlwgt",
        "final-sample-weight": "fnlwgt",
        "work-fnl": "fnlwgt",
    }
    renamed = []
    for col in df_clean.columns:
        raw = str(col).strip().lstrip("\ufeff")
        key = re.sub(r"[\s_]+", "-", raw.lower())
        renamed.append(semantic.get(key, key))
    duplicates = pd.Series(renamed).duplicated(keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate column names after standardizing headers: "
            f"{sorted({n for n, dup in zip(renamed, duplicates) if dup})}"
        )
    df_clean.columns = renamed

    # 1. Remove Duplicates
    df_clean = df_clean.drop_duplicates()

    # 2. Handle Missing Values (Adult-style column names after header normalization above)
    df_clean = df_clean.replace(" ?", np.nan)
    df_clean = df_clean.replace("?", np.nan)
    key_cols = [c for c in ("workclass", "occupation") if c in df_clean.columns]
    if len(key_cols) < 2:
        raise KeyError(
            "Expected columns 'workclass' and 'occupation' after header normalization. "
            f"Got: {list(df_clean.columns)}"
        )
    df_clean = df_clean.dropna(subset=key_cols)
    country_mode = df_clean["native-country"].mode(dropna=True)
    if len(country_mode):
        df_clean["native-country"] = df_clean["native-country"].fillna(
            country_mode.iloc[0]
        )

    # 3. Handle Outliers (IQR)
    cols_to_cap = ["age", "hours-per-week"]
    for col in cols_to_cap:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    # 4. Clean Text & Drop Redundant Columns
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].str.strip()

    if "education" in df_clean.columns:
        df_clean = df_clean.drop(columns=["education"])

    return df_clean





#phase 2
def preprocessing_phase_2(X_train, X_test, y_train, y_test):

    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    # 1. Encode Target
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)  # TRANSFORM ONLY!

    # 2. Encode Categorical Features
    mapping_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    for col in mapping_cols:
        le = LabelEncoder()
        # Fit & transform on training data
        X_train_processed[col] = le.fit_transform(X_train_processed[col])
        # Transform validation data using the rules learned from training data
        X_test_processed[col] = le.transform(X_test_processed[col])

    # 3. Feature Scaling
    scaler = StandardScaler()
    numeric_requested = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    train_cols = set(X_train_processed.columns)
    test_cols = set(X_test_processed.columns)
    absent_train = sorted(set(numeric_requested) - train_cols)
    absent_test = sorted(set(numeric_requested) - test_cols)
    if absent_train or absent_test:
        raise KeyError(
            "Expected these numeric columns for scaling (after Phase 1 header normalization): "
            f"{numeric_requested}. Missing in X_train: {absent_train}; missing in X_test: {absent_test}. "
            f"X_train columns now: {sorted(train_cols)}."
        )
    num_cols = numeric_requested

    # .fit_transform calculates the mean/std on X_train and scales it
    X_train_processed[num_cols] = scaler.fit_transform(X_train_processed[num_cols])
    # .transform uses the mean/std from X_train to scale X_test
    X_test_processed[num_cols] = scaler.transform(X_test_processed[num_cols])

    return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded