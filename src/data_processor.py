import pandas as pd
import matplotlib.pyplot as plt

def load_and_explore_data(filepath):
    """Loads data and performs initial exploration."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {filepath}")

    print("First 5 rows:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nDescription:")
    print(df.describe())
    print("\nInfo:")
    print(df.info())
    
    plt.figure(figsize=(15, 5))
    plt.plot(df['Close'])
    plt.title('Bitcoin Close price.', fontsize=15)
    plt.ylabel('Price in dollars.')
    plt.show()
    
    return df

def clean_data(df):
    """Cleans the dataframe by removing redundant columns and checking for nulls."""
    # Check if Close and Adj Close are the same
    if 'Adj Close' in df.columns:
        print(f"\nRows where Close == Adj Close shape: {df[df['Close'] == df['Adj Close']].shape}")
        print(f"Original shape: {df.shape}")
        df = df.drop(['Adj Close'], axis=1)
    
    print("\nNull values count:")
    print(df.isnull().sum())
    return df