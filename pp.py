import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
data = {
    'age': np.random.normal(loc=30, scale=10, size=1000).astype(int),
    'salary': np.random.normal(loc=50000, scale=15000, size=1000).astype(int),
    'experience': np.random.normal(loc=5, scale=2, size=1000).astype(int)
}
df = pd.DataFrame(data)

mean_values = df.mean()
median_values = df.median()
mode_values = df.mode().iloc[0]

print("Mean:\n", mean_values)
print("\nMedian:\n", median_values)
print("\nMode:\n", mode_values)

sns.set(style="whitegrid")
plt.figure(figsize=(16, 10))

plt.subplot(3, 1, 1)
sns.histplot(df['age'], kde=True, color='skyblue')
plt.axvline(mean_values['age'], color='red', linestyle='--', label='Mean')
plt.axvline(median_values['age'], color='green', linestyle='--', label='Median')
plt.axvline(mode_values['age'], color='orange', linestyle='--', label='Mode')
plt.title('Age Distribution')
plt.legend()

plt.subplot(3, 1, 2)
sns.histplot(df['salary'], kde=True, color='salmon')
plt.axvline(mean_values['salary'], color='red', linestyle='--', label='Mean')
plt.axvline(median_values['salary'], color='green', linestyle='--', label='Median')
plt.axvline(mode_values['salary'], color='orange', linestyle='--', label='Mode')
plt.title('Salary Distribution')
plt.legend()

plt.subplot(3, 1, 3)
sns.histplot(df['experience'], kde=True, color='lightgreen')
plt.axvline(mean_values['experience'], color='red', linestyle='--', label='Mean')
plt.axvline(median_values['experience'], color='green', linestyle='--', label='Median')
plt.axvline(mode_values['experience'], color='orange', linestyle='--', label='Mode')
plt.title('Experience Distribution')
plt.legend()

plt.tight_layout()
plt.show()
