import pandas as pd
import numpy as np

def calculate_sma(series, window=14):
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()

def calculate_ema(series, span=14):
    """Calculates Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, window=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # Fill NaN with neutral 50

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD and Signal line."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line