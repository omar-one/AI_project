"""Train baseline models and tuned decision tree."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import DT_PARAM_GRID, RANDOM_STATE


def train_baseline_models(X_train, y_train, X_test):
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    svm = SVC(kernel="rbf", random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    return {
        "lr": lr,
        "lr_pred": lr_pred,
        "svm": svm,
        "svm_pred": svm_pred,
        "dt": dt,
        "dt_pred": dt_pred,
    }


def tune_decision_tree(X_train, y_train, X_test):
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        DT_PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_dt = grid_search.best_estimator_
    best_dt_pred = best_dt.predict(X_test)
    return {
        "grid_search": grid_search,
        "best_dt": best_dt,
        "best_dt_pred": best_dt_pred,
    }
