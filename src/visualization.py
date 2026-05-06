import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv('../data/train.csv')

# First Visualization
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Second Visualization: Percentage Salary.
salary_percent = df['salary'].value_counts(normalize=True) * 100
sns.barplot(x=salary_percent.index, y=salary_percent.values)
plt.title('Salary Distribution (Percentage)')
plt.ylabel('Percentage %')
plt.xlabel('Salary Category')
plt.show()

# Third Visualization
position_percent = pd.crosstab(df['position'], df['salary'], normalize='index') * 100
position_percent.plot(kind='barh', stacked=True, figsize=(10, 8))
plt.title('Position vs Salary Percentage')
plt.xlabel('Percentage %')
plt.ylabel('Position')
plt.legend(title='Salary', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Forth Visualization
country_salary = pd.crosstab(
    df['native-country'],
    df['salary'],
    normalize='index'
) * 100
high_salary_col = [col for col in country_salary.columns if '>50K' in col][0]
country_salary = country_salary.sort_values(by=high_salary_col, ascending=True)
sns.set_style("whitegrid")
plt.figure(figsize=(10, 12))
ax = country_salary.plot(
    kind='barh',
    stacked=True,
    figsize=(10, 12),
    color=['#3498db', '#e67e22'], # Soft Blue and Orange
    width=0.8
)
plt.title('Percentage of Salary Levels by Native Country', fontsize=16, pad=20)
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('Native Country', fontsize=12)
plt.legend(title='Salary', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Fifth Visualization - FIXED
df_50 = df[df['age'] > 50]

# Standardize the salary column to remove whitespace and ensure correct casing
df_50['salary'] = df_50['salary'].str.strip()

total_by_dept = df_50.groupby('position').size()

# Fix: Use capital 'K' to match the dataset labels
high_salary = df_50[df_50['salary'] == '>50K']

high_by_dept = high_salary.groupby('position').size()

# Calculate percentage and handle missing values
percentage = (high_by_dept / total_by_dept) * 100
percentage = percentage.fillna(0)

# Plotting
plt.figure(figsize=(10, 8))
percentage.sort_values().plot(kind='barh', color='skyblue')

plt.xlabel('Percentage of Employees >50 earning >50K (%)')
plt.title('High Salary Rate for Employees Above 50 by Position')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
