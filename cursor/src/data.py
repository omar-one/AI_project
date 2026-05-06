"""Load dataset and build train/test splits."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TARGET, TEST_SIZE
from preprocessing import preprocessing_phase_1, split_train_test


def load_dataset(data_path: str) -> pd.DataFrame:
    """Read CSV; skipinitialspace normalizes values like ' Private' -> 'Private'."""
    return pd.read_csv(data_path, skipinitialspace=True)


def split_train_test(
    df: pd.DataFrame,
    *,
    target: str = TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
