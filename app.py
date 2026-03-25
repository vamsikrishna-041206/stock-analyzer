from flask import Flask, request, render_template
import yfinance as yf
import requests
import re
from datetime import date
import os

app = Flask(__name__)

# --- Configuration ---
# Use Environment Variables for security in the cloud, but allow a fallback for local testing
AV_API_KEY = os.environ.get('AV_API_KEY', 'PKO2Q4UFF8MN1DA8')
AV_BASE_URL = 'https://www.alphavantage.co/query'
AV_DAILY_LIMIT = 25

# --- In-Memory Rate Limiter ---
class InMemoryRateLimiter:
    def __init__(self, daily_limit):
        self.daily_limit = daily_limit
        self.date = str(date.today())
        self.count = 0

    def can_make_request(self):
        today = str(date.today())
        if self.date != today:
            self.date = today
            self.count = 0
        return self.count < self.daily_limit

    def increment(self):
        self.count += 1
        print(f"  [API Usage] {self.count}/{self.daily_limit} calls used today.")

# Global instance of the rate limiter
limiter = InMemoryRateLimiter(AV_DAILY_LIMIT)

# --- Sentiment Analyzer ---
class BasicSentimentAnalyzer:
    def __init__(self):
        self.positive_words = {'growth', 'grow', 'surge', 'rise', 'up', 'gain', 'bull', 'buy', 'strong', 'profit', 'beat', 'record', 'high', 'jump', 'rally', 'positive', 'win', 'upgrade', 'dividend'}
        self.negative_words = {'drop', 'fall', 'down', 'loss', 'lose', 'miss', 'bear', 'sell', 'weak', 'low', 'crash', 'decline', 'negative', 'fail', 'debt', 'risk', 'downgrade', 'sue', 'cut'}

    def analyze(self, text):
        if not text: return 0.0
        words = re.findall(r'\w+', text.lower())
        if not words: return 0.0
        score = sum(1 for w in words if w in self.positive_words) - sum(1 for w in words if w in self.negative_words)
        return max(min(score / 5.0, 1.0), -1.0)

analyzer = BasicSentimentAnalyzer()

# --- Helper Functions ---
def get_symbol_from_name(query):
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        response = requests.get(url, params={'q': query, 'quotesCount': 1}, headers={'User-Agent': 'Mozilla/5.0'})
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes'][0]['symbol']
    except Exception as e:
        print(f"Error resolving {query}: {e}")
    return None

def analyze_stock(symbol):
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="6mo")
        
        if len(history) < 20:
            return {"symbol": symbol, "error": "Not enough historical data for SMA."}

        # 1. Price & SMA calculations (In-Memory)
        latest_price = float(history['Close'].iloc[-1])
        sma_20 = float(history['Close'].tail(20).mean())
        price_vs_sma_pct = (latest_price - sma_20) / sma_20

        # 2. Sentiment Analysis
        sentiment_score = 0
        sentiment_source = "None"

        # Try Alpha Vantage First
        if limiter.can_make_request():
            try:
                av_res = requests.get(AV_BASE_URL, params={'function': 'NEWS_SENTIMENT', 'tickers': symbol, 'apikey': AV_API_KEY}).json()
                if "feed" in av_res:
                    limiter.increment()
                    total_weight = 0
                    weighted_score = 0
                    for item in av_res['feed']:
                        ticker_data = next((t for t in item.get('ticker_sentiment', []) if t['ticker'] == symbol), None)
                        score = float(ticker_data['ticker_sentiment_score']) if ticker_data else float(item.get('overall_sentiment_score', 0))
                        relevance = float(ticker_data['relevance_score']) if ticker_data else 0.5
                        weighted_score += score * relevance
                        total_weight += relevance
                    if total_weight > 0:
                        sentiment_score = weighted_score / total_weight
                        sentiment_source = "Alpha Vantage"
            except Exception as e:
                print(f"AV Error: {e}")

        # Fallback to Local YFinance News
        if sentiment_source == "None":
            try:
                news = ticker.news
                if news:
                    total_score = sum(analyzer.analyze(item.get('title', '')) for item in news)
                    sentiment_score = total_score / len(news)
                    sentiment_source = "Local (YFinance News)"
            except Exception as e:
                print(f"Local News Error: {e}")

        # 3. Decision Logic
        signal = "HOLD"
        rationale = "Mixed signals."
        
        # Adjust thresholds based on source
        bull_thresh = 0.15 if sentiment_source == "Alpha Vantage" else 0.1
        bear_thresh = -0.15 if sentiment_source == "Alpha Vantage" else -0.1
        
        if price_vs_sma_pct > 0 and sentiment_score > 0:
            signal = "STRONG BUY" if (price_vs_sma_pct > 0.02 and sentiment_score > bull_thresh) else "BUY"
            rationale = "Price is above SMA and sentiment is positive."
        elif price_vs_sma_pct < 0 and sentiment_score < 0:
            signal = "STRONG SELL" if (price_vs_sma_pct < -0.02 and sentiment_score < bear_thresh) else "SELL"
            rationale = "Price is below SMA and sentiment is negative."

        return {
            "symbol": symbol,
            "latest_price": round(latest_price, 2),
            "sma_20": round(sma_20, 2),
            "price_vs_sma_pct": round(price_vs_sma_pct * 100, 2),
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_source": sentiment_source,
            "signal": signal,
            "rationale": rationale
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# --- Web Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    results = []
    if request.method == 'POST':
        user_input = request.form.get('symbols', '')
        raw_inputs = [s.strip() for s in user_input.split(',') if s.strip()]
        
        for query in raw_inputs:
            symbol = get_symbol_from_name(query) or query.upper() # Fallback to raw input if search fails
            analysis = analyze_stock(symbol)
            results.append(analysis)
            
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)