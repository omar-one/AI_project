"""Metrics and feature importance reporting."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate(name: str, y_true, y_pred) -> float:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["<=50K", ">50K"],
        )
    )
    return acc


def feature_importance_df(X_columns, decision_tree) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "Feature": X_columns,
            "Importance": decision_tree.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
    )
