### Bitcoin Price Analysis, Prediction & Forecasting

This project implements a **complete machine learning and deep learning pipeline** for analyzing and predicting Bitcoin price movements using historical market data.

The project addresses two core problems:

1. **Price Direction Prediction (Classification)**
2. **Price Forecasting (Regression)** 

### Key Components

### 1. Data Loading & Exploration

- Loads historical Bitcoin OHLC price data
- Performs initial inspection
- Visualizes long-term closing price trends

### 2. Data Cleaning

- Removes redundant columns
- Checks and handles missing values
- Ensures data consistency for time-series modeling

### 3. Feature Engineering

- Date decomposition
- Financial behavior features:
    - Open–Close spread
    - Low–High spread
    - Quarter-end indicator
- **Technical indicators**:
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - Relative Strength Index (RSI)
    - MACD and Signal line
- Target variable creation:
    - Binary label indicating next-day price increase/decrease

### 4. Exploratory Data Analysis (EDA)

- Distribution and boxplot analysis of engineered features
- Correlation heatmaps to detect multicollinearity
- Visualization of yearly price behavior

### 5. Machine Learning Models (Classification)

Used to predict **price direction**:
- Logistic Regression
- Support Vector Machine (Polynomial Kernel)
- XGBoost Classifier

Evaluation:
- ROC-AUC score
- Train vs validation comparison
- Confusion matrix visualization

### 6. Deep Learning Forecasting (LSTM)
- Sequence-based time-series modeling using LSTM
- Min-Max scaling for stable training
- **TimeSeriesSplit backtesting** to respect temporal order
- Performance metrics:
    - RMSE
    - MAE
- Visualization of predicted vs actual prices for each fold

### Tech Stack

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **XGBoost**
- **TensorFlow / Keras (LSTM)**

## Dataset
This project uses the Bitcoin Price data- OHLC('Open', 'High', 'Low', 'Close') data from 17th July 2014 to 29th December 2022 which is for 8 years for the Bitcoin price. 
You can download it [here](https://media.geeksforgeeks.org/wp-content/uploads/20240917132611/bitcoin.csv).
