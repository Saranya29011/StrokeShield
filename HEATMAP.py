import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your dataset is in a CSV file named 'dataset.csv'
df = pd.read_csv("C:/Users/saran/Downloads/healthcare-dataset-stroke-data (1).csv")

# Convert categorical columns to numerical using one-hot encoding
df_encoded = pd.get_dummies(df)

# Compute the correlation matrix
correlation_matrix = df_encoded.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
