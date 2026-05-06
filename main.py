import sys
from pathlib import Path

import pandas as pd

# Treat `src/` as the import root so this script can be run as `python main.py`
# without installing a package or using `python -m`.
_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC))

from consts import DATA_PATH_TRAIN, DATA_PATH_TEST
from evaluation import evaluate
from preprocessing import preprocessing_phase_1, preprocessing_phase_2, split_train_test
from training_models import (
    tune_LogisticRegression,
    tune_RandomForest,
    tune_decision_tree,
    tune_svm,
)


print("Loading dataset...")

df_train = pd.read_csv(DATA_PATH_TRAIN, skipinitialspace=True)
df_test = pd.read_csv(DATA_PATH_TEST, skipinitialspace=True)
print(f"Dataset shape: {df_train.shape}")
print(f"Dataset shape: {df_test.shape}")

print("\nStarting Preprocessing Phase 1 (cleaning)...")
df_train = preprocessing_phase_1(df_train)
df_test = preprocessing_phase_1(df_test)

X_train, y_train = split_train_test(df_train)
X_test, y_test = split_train_test(df_test)

print("\nStarting Preprocessing Phase 2 (encoding and scaling)...")
X_train, X_test, y_train, y_test = preprocessing_phase_2(
    X_train, X_test, y_train, y_test
)
print("Preprocessing completed successfully!")

print("\nTraining models...")
lr_results = tune_LogisticRegression(X_train, y_train)
print(lr_results)
svm_results = tune_svm(X_train, y_train)
print(svm_results)
dt_results = tune_decision_tree(X_train, y_train)
print(dt_results)
rf_results = tune_RandomForest(X_train, y_train)
print(rf_results)

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)
lr_pred = lr_results["estimator"].predict(X_test)
svm_pred = svm_results["estimator"].predict(X_test)
dt_pred = dt_results["estimator"].predict(X_test)
rf_pred = rf_results["estimator"].predict(X_test)

evaluate("Logistic Regression", y_test, lr_pred)
evaluate("SVM", y_test, svm_pred)
evaluate("Default Decision Tree", y_test, dt_pred)
evaluate("Random Forest", y_test, rf_pred)


# fi = feature_importance_df(X_train.columns, results["best_dt"])
# print("\nTop 10 Most Important Features:")
# print(fi.head(10))

# print("\nProject completed successfully.")
# print(f"Best Decision Tree Accuracy: {dt_acc:.4f}")