import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from news_sentiment import EgxNewsSentiment


class EgxPredictor:
    def __init__(self, symbol="COMI.CA", target_horizon=1):
        """
        :param symbol: EGX ticker symbol (e.g. COMI, TMGH, ISPH)
        :param target_horizon: Number of trading days ahead to predict movement (1, 3, or 5 days)
        """
        symbol_upper = symbol.strip().upper()
        if not symbol_upper.endswith(".CA"):
            symbol_upper += ".CA"
        self.symbol = symbol_upper
        self.target_horizon = target_horizon
        self.df = None
        self.model = None
        self.sentiment_info = None

    def fetch_data(self, period="5y"):
        ticker_obj = yf.Ticker(self.symbol)
        self.df = ticker_obj.history(period=period)
        if self.df.empty:
            raise ValueError(f"No price data found for symbol '{self.symbol}'. The stock may be delisted or invalid.")
        print(f"Successfully fetched data for {self.symbol}")
        print(f"Downloaded {len(self.df)} daily records.")

    def add_technical_indicators(self):
        df = self.df.copy()
        
        # Returns and Lagged Returns
        df['Returns'] = df['Close'].pct_change()
        df['Return_Lag1'] = df['Returns'].shift(1)
        df['Return_Lag2'] = df['Returns'].shift(2)
        df['Return_Lag3'] = df['Returns'].shift(3)
        df['Return_Lag5'] = df['Returns'].shift(5)

        # Moving Averages & Ratios
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['Price_to_MA20'] = df['Close'] / df['MA20']
        df['Price_to_MA50'] = df['Close'] / df['MA50']

        # Volatility
        df['Volatility20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['MA20'] + (bb_std * 2)
        df['BB_Lower'] = df['MA20'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20']
        df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)

        # Volume Change & Ratios
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA5'] + 1e-9)

        # Fetch News Sentiment Feature
        print("Fetching news sentiment...")
        sentiment_analyzer = EgxNewsSentiment(self.symbol)
        self.sentiment_info = sentiment_analyzer.get_market_sentiment()
        df['Sentiment_Score'] = self.sentiment_info['score']

        # Target: 1 if Close N days ahead > today's Close, else 0
        df['Target'] = (df['Close'].shift(-self.target_horizon) > df['Close']).astype(int)

        # Replace infinite values resulting from division by zero with NaN and drop them
        df = df.replace([np.inf, -np.inf], np.nan)
        self.df = df.dropna()

    def get_feature_names(self):
        return [
            'Returns', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag5',
            'MA5', 'MA10', 'MA20', 'MA50', 'Price_to_MA20', 'Price_to_MA50',
            'Volatility20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Width', 'BB_Pos', 'Volume_Change', 'Volume_Ratio', 'Sentiment_Score'
        ]

    def train_model(self):
        """Train XGBoost model on technical features with class weighting."""
        features = self.get_feature_names()
        X = self.df[features]
        y = self.df['Target']

        # Sequential train-test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Calculate positive class weighting to handle class imbalance
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        pos_weight = num_neg / max(num_pos, 1)

        print(f"Training XGBoost Classifier ({self.target_horizon}-Day Horizon) on {len(X_train)} samples with {len(features)} features...")
        
        self.model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=pos_weight,
            random_state=42,
            min_child_weight=3
        )
        self.model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy on Test Set ({self.target_horizon}-Day Horizon): {acc * 100:.2f}%")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))

    def save_model(self, models_dir="models"):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("Model is not trained yet. Call train_model() first.")
        os.makedirs(models_dir, exist_ok=True)
        filepath = os.path.join(models_dir, f"{self.symbol}_{self.target_horizon}d_xgb.joblib")
        joblib.dump(self.model, filepath)
        print(f"Saved trained model to {filepath}")
        return filepath

    def load_model(self, models_dir="models"):
        """Load trained model from file."""
        filepath = os.path.join(models_dir, f"{self.symbol}_{self.target_horizon}d_xgb.joblib")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at {filepath}")
        self.model = joblib.load(filepath)
        print(f"Loaded trained model from {filepath}")

    def predict_tomorrow(self):
        """Predict movement for the target horizon."""
        features = self.get_feature_names()
        latest_features = self.df[features].iloc[[-1]]

        prediction = self.model.predict(latest_features)[0]
        probabilities = self.model.predict_proba(latest_features)[0]
        direction = "[UP]" if prediction == 1 else "[DOWN/FLAT]"
        confidence = max(probabilities) * 100

        print("=" * 50)
        print(f"PREDICTION FOR {self.symbol} ({self.target_horizon}-DAY HORIZON):")
        print(f"Signal: {direction}")
        print(f"Confidence: {confidence:.2f}%")
        if self.sentiment_info:
            print(f"News Sentiment: {self.sentiment_info['label']} (Score: {self.sentiment_info['score']})")
        print("=" * 50)
        
        return {
            "symbol": self.symbol,
            "target_horizon": self.target_horizon,
            "prediction": int(prediction),
            "signal": direction,
            "confidence": round(confidence, 2),
            "sentiment": self.sentiment_info
        }


if __name__ == "__main__":
    user_input = input("Enter stock symbol (e.g. COMI, TMGH, ISPH, RAYA): ").strip()
    symbol_name = user_input if user_input else "COMI"
    horizon_input = input("Enter target horizon in trading days (1, 3, 5) [default=1]: ").strip()
    target_h = int(horizon_input) if horizon_input.isdigit() else 1
    
    try:
        predictor = EgxPredictor(symbol=symbol_name, target_horizon=target_h)
        predictor.fetch_data(period="5y")
        predictor.add_technical_indicators()
        predictor.train_model()
        predictor.save_model()
        predictor.predict_tomorrow()
    except Exception as e:
        print(f"\nError running prediction: {e}")