import yfinance as yf
from textblob import TextBlob
import re

class EgxNewsSentiment:
    def __init__(self, symbol="COMI.CA"):
        clean_symbol = symbol.strip().upper()
        if not clean_symbol.endswith(".CA"):
            clean_symbol += ".CA"
        self.symbol = clean_symbol

    def fetch_recent_news(self):
        """Fetch recent news headlines from yfinance for the ticker."""
        try:
            ticker = yf.Ticker(self.symbol)
            news_items = ticker.news
            headlines = []
            if news_items:
                for item in news_items:
                    title = ""
                    if isinstance(item, dict):
                        content = item.get("content", {})
                        if isinstance(content, dict):
                            title = content.get("title", "")
                        if not title:
                            title = item.get("title", "")
                    if title:
                        headlines.append(title)
            return headlines
        except Exception as e:
            print(f"Warning: Could not fetch news for {self.symbol}: {e}")
            return []

    def analyze_headline_sentiment(self, text):
        """Calculate sentiment polarity (-1.0 to +1.0) using TextBlob & keyword weights."""
        if not text:
            return 0.0
        
        # Financial domain keyword adjustments
        text_lower = text.lower()
        bullish_keywords = ["profit", "growth", "dividend", "revenue", "surge", "gain", "expand", "acquisition", "record", "upgrade"]
        bearish_keywords = ["loss", "drop", "decline", "fall", "debt", "slash", "cut", "downgrade", "crisis", "plunge"]
        
        score = TextBlob(text).sentiment.polarity
        
        for word in bullish_keywords:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                score += 0.2
        for word in bearish_keywords:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                score -= 0.2

        # Clamp between -1.0 and +1.0
        return max(-1.0, min(1.0, score))

    def get_market_sentiment(self):
        headlines = self.fetch_recent_news()
        if not headlines:
            return {
                "score": 0.0,
                "label": "NEUTRAL",
                "headlines": ["No recent news headlines available for this ticker."]
            }

        scores = [self.analyze_headline_sentiment(h) for h in headlines]
        avg_score = sum(scores) / len(scores)

        if avg_score > 0.1:
            label = "BULLISH"
        elif avg_score < -0.1:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return {
            "score": round(avg_score, 3),
            "label": label,
            "headlines": headlines[:5]
        }

if __name__ == "__main__":
    sentiment_analyzer = EgxNewsSentiment("COMI.CA")
    result = sentiment_analyzer.get_market_sentiment()
    print("Market Sentiment Result:")
    print(f"Score: {result['score']}")
    print(f"Label: {result['label']}")
    print("Headlines:", result['headlines'])
