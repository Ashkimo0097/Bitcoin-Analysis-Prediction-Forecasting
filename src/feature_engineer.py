import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

try:
    from src.technical_indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd
except ImportError:
    try:
        from .technical_indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd
    except ImportError:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.technical_indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd

def feature_engineering(df):
    """Performs feature engineering including date splitting and creating new features."""
    # Date splitting
    splitted = df['Date'].str.split('-', expand=True)
    df['year'] = splitted[0].astype('int')
    df['month'] = splitted[1].astype('int')
    df['day'] = splitted[2].astype('int')

    # Convert 'Date' to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Grouped data visualization (Side effect in feature engineering, consider moving to visualizer ideally, but keeping logic flow)
    data_grouped = df.groupby('year').mean(numeric_only=True)
    plt.subplots(figsize=(20, 10))
    for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
        plt.subplot(2, 2, i + 1)
        data_grouped[col].plot.bar()
    plt.show()

    # Create derived features
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']
    
    # Add Technical Indicators
    df['sma_7'] = calculate_sma(df['Close'], window=7)
    df['sma_30'] = calculate_sma(df['Close'], window=30)
    df['ema_12'] = calculate_ema(df['Close'], span=12)
    df['rsi'] = calculate_rsi(df['Close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['Close'])
    
    # Fill any NaNs created by rolling windows to avoid errors in training
    df = df.fillna(method='bfill').fillna(method='ffill')

    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    return df
    