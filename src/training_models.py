from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import RANDOM_STATE

def tune_decision_tree(X_train, y_train):
    dt_param_space = {
        "max_depth": Integer(5, 50),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2", 0.3, 0.5, 0.7]),
        "criterion": Categorical(["gini", "entropy"]),
    }

    dt_bayes_search = BayesSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        dt_param_space,
        n_iter=30,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    dt_bayes_search.fit(X_train, y_train)
    best_dt = dt_bayes_search.best_estimator_

    return {
        "estimator": best_dt,
        "search": dt_bayes_search,
        "best_params": dt_bayes_search.best_params_,
        "best_score": dt_bayes_search.best_score_,
    }

def tune_svm(X_train, y_train):
    svc_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly'],
    }

    svc_grid_search = GridSearchCV(
        SVC(random_state=RANDOM_STATE),
        svc_param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    svc_grid_search.fit(X_train, y_train)
    best_svm = svc_grid_search.best_estimator_

    return {
        "estimator": best_svm,
        "search": svc_grid_search,
        "best_params": svc_grid_search.best_params_,
        "best_score": svc_grid_search.best_score_,
    }

def tune_LogisticRegression(X_train, y_train):
    lr_param_space = {
        'C': Real(0.01, 10, prior='log-uniform'),
        'max_iter': Categorical([500, 1000, 1500, 2000]),
        'solver': Categorical(['lbfgs', 'liblinear', 'saga'])
    }

    lr_bayes_search = BayesSearchCV(
        LogisticRegression(random_state=RANDOM_STATE),
        lr_param_space,
        n_iter=30,  # try 30 smart combinations
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    lr_bayes_search.fit(X_train, y_train)
    lr = lr_bayes_search.best_estimator_

    return {
        "estimator": lr,
        "search": lr_bayes_search,
        "best_params": lr_bayes_search.best_params_,
        "best_score": lr_bayes_search.best_score_,
    }

def tune_RandomForest(X_train, y_train):
    rf_param_space = {
        "n_estimators": Integer(100, 500),
        "max_depth": Integer(5, 50),
        "min_samples_split": Integer(2, 20),
        "max_features": Categorical(["sqrt", "log2", 0.3, 0.5, 0.7]),
    }

    rf_bayes_search = BayesSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        rf_param_space,
        n_iter=30,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    rf_bayes_search.fit(X_train, y_train)
    rf = rf_bayes_search.best_estimator_

    return {
        "estimator": rf,
        "search": rf_bayes_search,
        "best_params": rf_bayes_search.best_params_,
        "best_score": rf_bayes_search.best_score_,
    }