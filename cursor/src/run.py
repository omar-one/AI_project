"""CLI entry: load, preprocess, train, evaluate."""

import warnings

from .config import DATA_PATH
from .data import load_dataset, split_train_test
from .evaluation import evaluate, feature_importance_df
from .preprocessing import preprocess
from .training import train_baseline_models, tune_decision_tree

warnings.filterwarnings("ignore")


def main() -> None:
    print("Loading dataset...")

    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test = split_train_test(df)
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    print("\nStarting Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess(
        X_train, X_test, y_train, y_test
    )
    print("Preprocessing completed successfully!")

    print("\nTraining models...")
    results = train_baseline_models(X_train, y_train, X_test)

    print("\nTuning Decision Tree (this may take a minute)...")
    tuned = tune_decision_tree(X_train, y_train, X_test)
    results.update(tuned)
    print(
        "Best Parameters for Decision Tree: "
        f"{results['grid_search'].best_params_}"
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    evaluate("Logistic Regression", y_test, results["lr_pred"])
    evaluate("SVM", y_test, results["svm_pred"])
    evaluate("Default Decision Tree", y_test, results["dt_pred"])
    dt_acc = evaluate(
        "Tuned Decision Tree (Best)",
        y_test,
        results["best_dt_pred"],
    )

    fi = feature_importance_df(X_train.columns, results["best_dt"])
    print("\nTop 10 Most Important Features:")
    print(fi.head(10))

    print("\nProject completed successfully.")
    print(f"Best Decision Tree Accuracy: {dt_acc:.4f}")
