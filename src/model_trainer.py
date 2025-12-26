import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

def train_and_evaluate(df):
    """Trains models and evaluates them."""
    features = df[['open-close', 'low-high', 'is_quarter_end', 'sma_7', 'sma_30', 'ema_12', 'rsi', 'macd', 'macd_signal']]
    target = df['target']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        features_scaled, target, test_size=0.3, random_state=42
    )

    models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

    for model in models:
        model.fit(X_train, Y_train)
        print(f'{model} : ')
        
        train_acc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
        valid_acc = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1])
        
        print(f'Training Accuracy : {train_acc}')
        print(f'Validation Accuracy : {valid_acc}')
        print()

    # Confusion Matrix for the first model
    ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid, cmap='Blues')
    plt.show()