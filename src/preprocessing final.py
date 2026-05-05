import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Data
df = pd.read_csv('E:\\hanako\\CS stuff\\uni\\Meterials\\Level2\\AI\\proj\\Train.csv')

# 2. Identify and Remove Duplicates
print(f"Duplicates found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# 3. Handle Missing Values
df.replace(' ?', np.nan, inplace=True)
# Drop rows for critical columns
df.dropna(subset=['work-class', 'position'], inplace=True)
# Fill missing values in 'native-country' with the mode
df['native-country'].fillna(df['native-country'].mode()[0], inplace=True)

# 4. Handle Outliers
# Manual capping logic to protect the lower bound of part-time workers
df['age'] = df['age'].clip(lower=17, upper=75)
df['hours-per-week'] = df['hours-per-week'].clip(upper=98)

# 5. Clean text data and drop redundant columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].str.strip()

# Drop 'education' since 'education-num' already encodes it perfectly
df.drop(columns=['education'], inplace=True)

# 6. Encode Categorical Data
le = LabelEncoder()

# Encode target variable
df['salary'] = le.fit_transform(df['salary'])

# Encode categorical features
mapping_cols = ['work-class', 'marital-status', 'position', 'relationship', 'race', 'sex', 'native-country']
for col in mapping_cols:
    df[col] = le.fit_transform(df[col])

# 7. Scaling and Normalization
# Ensures large numbers (like work-fnl) don't dominate smaller numbers (like age)
scaler = StandardScaler()
num_cols = ['age', 'work-fnl', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df[num_cols] = scaler.fit_transform(df[num_cols])

print("Preprocessing complete! Dataset is ready for classification models.")
print(df.head())