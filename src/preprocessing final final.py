import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from __future__ import annotations
from .config import RANDOM_STATE, TARGET, TEST_SIZE



# cleaning function
def preprocessing_phase_1(df):

    df_clean = df.copy()

    # 1. Remove Duplicates
    df_clean.drop_duplicates(inplace=True)

    # 2. Handle Missing Values
    df_clean.replace(' ?', np.nan, inplace=True)
    df_clean.dropna(subset=['work-class', 'position'], inplace=True)
    df_clean['native-country'].fillna(df_clean['native-country'].mode()[0], inplace=True)

    # 3. Handle Outliers (IQR)
    cols_to_cap = ['age', 'hours-per-week']

    for col in cols_to_cap:
        # Calculate Q1, Q3, and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        # Define boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap outliers using clip
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # 4. Clean Text & Drop Redundant Columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].str.strip()

    df_clean.drop(columns=['education'], inplace=True)

    return df_clean


# splitting function
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

#encoding and scaling function
def preprocessing_phase_2(X_train, X_val, y_train, y_val):

    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()

    # 1. Encode Target
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_val_encoded = le_target.transform(y_val)  # TRANSFORM ONLY!

    # 2. Encode Categorical Features
    mapping_cols = ['work-class', 'marital-status', 'position', 'relationship', 'race', 'sex', 'native-country']
    for col in mapping_cols:
        le = LabelEncoder()
        # Fit on training data, transform training data
        X_train_processed[col] = le.fit_transform(X_train_processed[col])
        # Transform validation data using the rules learned from training data
        X_val_processed[col] = le.transform(X_val_processed[col])

    # 3. Feature Scaling
    scaler = StandardScaler()
    num_cols = ['age', 'work-fnl', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # .fit_transform calculates the mean/std on X_train and scales it
    X_train_processed[num_cols] = scaler.fit_transform(X_train_processed[num_cols])
    # .transform uses the mean/std from X_train to scale X_val
    X_val_processed[num_cols] = scaler.transform(X_val_processed[num_cols])

    return X_train_processed, X_val_processed, y_train_encoded, y_val_encoded


# main function
if __name__ == "__main__":
    # 1. Load Data
    raw_df = pd.read_csv('pathx.csv', skipinitialspace=True)

    # 2. Phase 1: Clean
    cleaned_df = preprocessing_phase_1(raw_df)

    # 3. Split
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = split_data(cleaned_df)

    # 4. Phase 2: Encode & Scale
    X_train, X_val, y_train, y_val = preprocessing_phase_2(X_train_raw, X_val_raw, y_train_raw, y_val_raw)

    print("Execution Complete!")
    print(f"Final X_train shape ready for modeling: {X_train.shape}")