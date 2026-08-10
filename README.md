# 📈 EGX Stock Market Prediction System

A **Hybrid AI Stock Market Prediction System** for the Egyptian Exchange (EGX) that combines XGBoost machine learning with financial news sentiment analysis — featuring a live interactive web dashboard.

---

## ✨ Features

- 🤖 **XGBoost ML Engine** — Trained on 21 technical indicators per stock
- 📰 **News Sentiment Analysis** — Fetches real headlines and scores market bias (Bullish / Neutral / Bearish)
- 💡 **Buy / Hold / Sell Recommendations** — Smart signal engine combining prediction confidence + sentiment
- 🌍 **EGX Market Screener** — Scans ~40 EGX stocks and ranks them by buy/sell signal strength
- 📉 **Interactive Charts** — Candlestick, Bollinger Bands, MA20/50, RSI (14), and Volume
- 📊 **Feature Importance** — See which indicators drove each prediction
- ⚡ **Fully Reactive UI** — Results update automatically when you change any control (no button needed)
- 💾 **Model Persistence** — Trained models are saved and reloaded via `joblib`

---

## 🗂️ Project Structure

```
stock market prediction/
│
├── app.py               # Streamlit web dashboard (main entry point)
├── egx_predictor.py     # Core ML engine: data fetching, features, XGBoost, predictions
├── news_sentiment.py    # News headline fetcher & TextBlob sentiment scorer
├── main.py              # CLI batch pipeline for multiple tickers
│
├── models/              # Auto-created: saved XGBoost .joblib model files
└── README.md            # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install yfinance pandas numpy matplotlib xgboost scikit-learn streamlit plotly joblib textblob
```

### 2. Launch the Web Dashboard

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

### 3. Run CLI Batch Predictions

```bash
python main.py
```

This runs predictions for a default watchlist: `COMI, TMGH, ISPH, RAYA, SWDY`

### 4. Run a Single Stock Prediction

```bash
python egx_predictor.py
```

Enter the EGX ticker code when prompted (e.g. `COMI`, `TMGH`, `ISPH`).

---

## 📊 Technical Indicators Used (21 Features)

| Category | Indicators |
|---|---|
| **Returns** | Daily Return, Lag 1, 2, 3, 5 days |
| **Moving Averages** | MA5, MA10, MA20, MA50, Price/MA20 ratio, Price/MA50 ratio |
| **Momentum** | RSI (14), MACD, MACD Signal, MACD Histogram |
| **Volatility** | Bollinger Band Width, Bollinger Band Position, 20-day Annualised Volatility |
| **Volume** | Volume Day Change, Volume/5-Day MA Ratio |
| **Sentiment** | News Polarity Score (TextBlob + domain keyword weighting) |

---

## 💡 Buy / Hold / Sell Logic

| Condition | Signal |
|---|---|
| Model predicts **UP** + Confidence ≥ 58% | 🟢 **BUY** |
| Model predicts **DOWN/FLAT** + Confidence ≥ 58% | 🔴 **SELL** |
| Confidence < 58% (mixed/weak signals) | 🟡 **HOLD** |

News sentiment (Bullish / Bearish / Neutral) modifies the recommendation reasoning.

---

## 🏦 Supported EGX Tickers

Use any EGX stock code — the `.CA` Yahoo Finance suffix is added automatically.

**Popular examples:**

| Company | Ticker |
|---|---|
| Commercial International Bank | `COMI` |
| Talaat Moustafa Group (TMG) | `TMGH` |
| Ibnsina Pharma | `ISPH` |
| Raya Holding | `RAYA` |
| El Sewedy Electric | `SWDY` |
| EFG Hermes | `HRHO` |
| Ezz Steel | `ESRS` |
| Fawry | `FWRY` |
| Abou Kir Fertilizers | `ABUK` |
| Eastern Company | `EAST` |

---

## 🔮 Prediction Horizons

| Horizon | What it predicts |
|---|---|
| **1 Trading Day** | Will tomorrow's close be higher than today's? |
| **3 Trading Days** | Will the close in 3 days be higher than today's? |
| **5 Trading Days** | Will the close in 5 days be higher than today's? |

Longer horizons (3–5 days) tend to reduce noise and improve accuracy.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Historical stock OHLCV data & news headlines |
| `xgboost` | Gradient boosted tree classifier |
| `scikit-learn` | Train/test split & evaluation metrics |
| `pandas` / `numpy` | Data processing |
| `streamlit` | Interactive web dashboard |
| `plotly` | Interactive candlestick & indicator charts |
| `textblob` | Natural language sentiment analysis |
| `joblib` | Model serialisation & persistence |

---

## ⚠️ Disclaimer

> This tool is for **educational and research purposes only**.
> AI-generated predictions are based on historical price patterns and news sentiment — they do not guarantee future performance.
> **Always do your own due diligence before making any investment decisions.**

---

## 👨‍💻 Author

Built as a hybrid quantitative + LLM pipeline for Egyptian Exchange (EGX) stock market analysis.
