from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


def evaluate(name: str, y_true, y_pred) -> float:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    ctr = classification_report(y_true, y_pred, target_names=["<=50K", ">50K"])
    print("****************  " + name + "  ****************")
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(ctr)
    print("confusion_matrix: " + str(cm))
    return acc


def feature_importance_df(X_columns, decision_tree) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "Feature": X_columns,
            "Importance": decision_tree.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
    )