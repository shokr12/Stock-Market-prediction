from egx_predictor import EgxPredictor

def run_prediction(symbol: str = "COMI", period: str = "5y", target_horizon: int = 1):
    """
    Run prediction pipeline for any given EGX stock symbol.
    """
    print(f"\n==========================================")
    print(f"   PREDICTION PIPELINE: {symbol.upper()} ({target_horizon}-DAY HORIZON)")
    print(f"==========================================")
    
    try:
        predictor = EgxPredictor(symbol=symbol, target_horizon=target_horizon)
        predictor.fetch_data(period=period)
        predictor.add_technical_indicators()
        predictor.train_model()
        predictor.save_model()
        result = predictor.predict_tomorrow()
        return result
    except Exception as e:
        print(f"Error predicting {symbol}: {e}")
        return None

if __name__ == "__main__":
    watchlist = ["COMI", "TMGH", "ISPH", "RAYA", "SWDY"]
    
    print("Welcome to EGX Hybrid Stock Prediction Pipeline!")
    print(f"Evaluating Watchlist: {', '.join(watchlist)}\n")
    
    # Run 1-day predictions
    for ticker in watchlist:
        run_prediction(ticker, period="5y", target_horizon=1)
