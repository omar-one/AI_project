
# preprocessing_phase_1 -> split_train_test -> preprocessing_phase_2 ->       training       ->       evaluation      -> end
#     df -> df                  df -> x,y           x,y -> x,y           model -> result           model -> info

import pandas as pd
from consts import DATA_PATH_TRAIN, DATA_PATH_TEST
from evaluation import evaluate
from preprocessing import preprocessing_phase_1, preprocessing_phase_2, split_train_test
from training_models import tune_LogisticRegression,tune_RandomForest,tune_decision_tree,tune_svm


print("Loading dataset...")

df_train = pd.read_csv(DATA_PATH_TRAIN)
df_test = pd.read_csv(DATA_PATH_TEST)
print(f"Dataset of train shape: {df_train.shape}")
print(f"Dataset of test shape: {df_test.shape}")

print("\nStarting Preprocessing")
df_train = preprocessing_phase_1(df_train)
df_test = preprocessing_phase_1(df_test)

X_train, y_train = split_train_test(df_train)
X_test, y_test = split_train_test(df_test)


X_train, X_test, y_train, y_test = preprocessing_phase_2(
    X_train, X_test, y_train, y_test
)
print("Preprocessing completed successfully!")

print("\nTraining models...")
lr_results = tune_LogisticRegression(X_train, y_train)
svm_results = tune_svm(X_train, y_train)
dt_results = tune_decision_tree(X_train, y_train)
rf_results = tune_RandomForest(X_train, y_train)

print("\nPreprocessing completed successfully!")

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
rf_acc = evaluate("Random Forest", y_test, rf_pred)

print("\n" + "=" * 60)
importance = rf_results["estimator"].feature_importances_
fi_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(fi_df.head(5))

print("\nProject completed successfully.")

print(f"Best Random Forest Accuracy: {rf_acc:.4f}")