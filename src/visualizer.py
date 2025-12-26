import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

def visualize_distributions(df, feature_cols):
    """Visualizes distributions and boxplots for given features."""
    plt.subplots(figsize=(20, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 2, i + 1)
        sn.distplot(df[col])
    plt.show()

    plt.subplots(figsize=(20, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 2, i + 1)
        sn.boxplot(df[col], orient='h')
    plt.show()

def visualize_correlations(df):
    """Visualizes target distribution and feature correlations."""
    plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
    plt.show()

    plt.figure(figsize=(10, 10))
    # Select numeric columns for correlation to avoid errors
    numeric_df = df.select_dtypes(include=[np.number])
    sn.heatmap(numeric_df.corr() > 0.9, annot=True, cbar=False)
    plt.show()
    