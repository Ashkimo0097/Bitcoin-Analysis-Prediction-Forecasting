import warnings
from src.data_processor import load_and_explore_data, clean_data
from src.visualizer import visualize_distributions, visualize_correlations
from src.feature_engineer import feature_engineering
from src.model_trainer import train_and_evaluate
from src.forecasting import train_and_evaluate_forecast

warnings.filterwarnings('ignore')

def main():
    filepath = 'bitcoin.csv'
    
    try:
        df = load_and_explore_data(filepath)
    except FileNotFoundError as e:
        print(e)
        return

    df = clean_data(df)
    
    initial_features = ['Open', 'High', 'Low', 'Close']
    visualize_distributions(df, initial_features)
    
    df = feature_engineering(df)
    visualize_correlations(df)
    
    # Classification Task
    print("\n--- Running Classification Models ---")
    train_and_evaluate(df)

    # Time Series Forecasting Task
    print("\n--- Running Time Series Forecasting ---")
    train_and_evaluate_forecast(df, seq_length=60, epochs=5)

if __name__ == "__main__":
    main()