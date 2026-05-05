from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from .preprocessing import DT_PARAM_GRID, RANDOM_STATE


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

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    return {
        "lr": lr,
        "lr_pred": lr_pred,
        "svm": svm,
        "svm_pred": svm_pred,
        "dt": dt,
        "dt_pred": dt_pred,
        "rf": rf,
        "rf_pred": rf_pred,
        "svc_best_params" : svc_grid_search.best_params
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


def tune_svm(X_train, y_train, X_test, y_test):
    svc_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.1, 1, 10, 100],
        'kernel': ['rbf'],
    }

    svc_grid_search = GridSearchCV(
        SVC(random_state=RANDOM_STATE),
        svc_param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    # تدريب البحث الشبكي
    svc_grid_search.fit(X_train, y_train)


    best_svm = svc_grid_search.best_estimator_

    svm_pred = best_svm.predict(X_test)


    svc_acc = accuracy_score(y_test, svm_pred)


    print(f"SVC Accuracy after Tuning: {svc_acc * 100:.2f}%")
    print(f"Best Parameters found: {svc_grid_search.best_params_}")

    return {
        "svc_best_params": svc_grid_search.best_params_,
        "svm_pred": svm_pred,
        "svm_best_score": svc_grid_search.best_score_ 
    }
