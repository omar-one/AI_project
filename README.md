# Income classification (Adult-style dataset)

This project trains and compares classical machine learning models to predict whether an individual’s **salary** is **≤50K** or **>50K** from census-style tabular features.

## What’s included

- **`main.py`** — Thin entry point; runs `src.run.main()`.
- **`src/`** — Pipeline code: config, loading/splitting, preprocessing, training, evaluation (see layout below).
- **`data/Test.csv`** — Labeled dataset used as the **only** data source. The pipeline performs an **80/20 stratified train/test split** so evaluation uses data the models did not train on.

## Dataset

**File:** `data/Test.csv`  
**Format:** CSV with a header row. Many fields have leading spaces after commas in the raw file; loading uses `skipinitialspace=True` so values normalize cleanly.

| Column | Role |
|--------|------|
| `age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country` | Features |
| `salary` | Target (`<=50K` / `>50K`) |

Missing values in feature columns may appear as `?`; these are converted to missing values and imputed (see below).

## Requirements

Install Python 3.8+ and dependencies:

```bash
pip install pandas numpy scikit-learn
```

## How to run

From the project root (the directory that contains `main.py` and the `data` folder):

```bash
python main.py
```

Training includes **5-fold grid search** for the decision tree, so a full run may take **on the order of tens of seconds to a few minutes** depending on your machine.

## Pipeline overview

1. **Load** — Read `data/Test.csv`.
2. **Split** — `train_test_split` with `test_size=0.2`, `random_state=42`, `stratify=y`.
3. **Preprocess**
   - Replace `?` with missing values.
   - **Categorical** columns: `SimpleImputer(strategy='most_frequent')`, then `LabelEncoder` per column (fit on train, transform test).
   - **Numeric** columns: `SimpleImputer(strategy='median')`.
   - **Target:** strip whitespace and map `<=50K` → `0`, `>50K` → `1`.
4. **Models**
   - **LogisticRegression** — `max_iter=1000`, `random_state=42`.
   - **SVC** — `kernel='rbf'`, `random_state=42`.
   - **DecisionTreeClassifier** — default settings except `random_state=42`.
   - **Tuned decision tree** — `GridSearchCV` with `cv=5`, `scoring='accuracy'`, `n_jobs=-1` over:
     - `max_depth`: `[10, 15, 20]`
     - `min_samples_split`: `[2, 5, 10]`
     - `min_samples_leaf`: `[1, 2, 4]`
5. **Evaluate** — Accuracy and classification report per model on the **test** split; top decision-tree features by importance.

## Output

The script prints dataset shapes, preprocessing confirmation, best grid-search parameters for the tree, **accuracy** and **precision/recall/F1** for each model, and the **top 10 features** by importance from the tuned decision tree.

## Notes

- **`LabelEncoder` is fit only on training data per column**, which avoids leakage from the test set into category encodings.
- **SVM** with an RBF kernel on one-hot–free label-encoded categoricals can behave poorly or slowly at large scale; the script keeps it for comparison with linear and tree models.

## Project layout

```text
AI_project/
├── main.py
├── src/
│   ├── __init__.py
│   ├── config.py       # paths, column lists, hyperparameter grids
│   ├── data.py         # load CSV, train_test_split
│   ├── preprocessing.py
│   ├── training.py     # baseline models + GridSearchCV decision tree
│   ├── evaluation.py   # metrics, feature importance table
│   └── run.py          # orchestration and console output
├── data/
│   └── Test.csv
└── README.md
```
