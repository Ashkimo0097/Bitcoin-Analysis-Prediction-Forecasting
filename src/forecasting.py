import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Try to import tensorflow, handle if not installed
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    Sequential = None
    print("Warning: TensorFlow is not installed. Forecasting will not work.")
    print("Please install it using: pip install tensorflow")

def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM training.
    X: data[i : i + seq_length]
    y: data[i + seq_length] (Next step prediction)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(input_shape):
    """Builds a standard LSTM model."""
    if Sequential is None:
        raise ImportError("TensorFlow not found.")
        
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_evaluate_forecast(df, seq_length=60, epochs=10, batch_size=32):
    """
    Trains LSTM model using Time Series Cross Validation.
    Predicts the 'Close' price.
    """
    if Sequential is None:
        print("Skipping forecasting due to missing TensorFlow.")
        return

    print("\n" + "="*50)
    print("STARTING LSTM TIME SERIES FORECASTING")
    print("="*50)

    # Use Close price for prediction
    data = df[['Close']].values
    
    # Scale data (LSTMs are sensitive to scale)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    # Input: Past 'seq_length' days -> Output: Next day Close price
    X, y = create_sequences(scaled_data, seq_length)
    
    # Time Series Cross Validation (Backtesting)
    # This respects temporal order (train on past, test on future)
    tscv = TimeSeriesSplit(n_splits=3)
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"\n--- Backtesting Fold {fold} ---")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Predict
        predictions = model.predict(X_test)
        
        # Inverse transform to get actual prices
        predictions_actual = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
        mae = mean_absolute_error(y_test_actual, predictions_actual)
        print(f"Fold {fold} RMSE: ${rmse:.2f}")
        print(f"Fold {fold} MAE: ${mae:.2f}")
        
        # Visualize
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual, label='Actual Price', color='blue')
        plt.plot(predictions_actual, label='Predicted Price', color='red', linestyle='--')
        plt.title(f'Bitcoin Price Prediction (LSTM) - Backtest Fold {fold}\nRMSE: ${rmse:.2f}')
        plt.xlabel('Time Steps')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        fold += 1