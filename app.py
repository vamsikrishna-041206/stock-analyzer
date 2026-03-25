from flask import Flask, request, render_template
import yfinance as yf
import requests
import re
from datetime import date
import os

app = Flask(__name__)

# --- Configuration ---
# You will add HF_API_KEY in your Render Environment Variables later
HF_API_KEY = os.environ.get('HF_API_KEY')
HF_API_URL = "https://api.huggingface.co/models/ProsusAI/finbert"

# --- Fallback Sentiment Analyzer (If API fails) ---
class BasicSentimentAnalyzer:
    def __init__(self):
        self.positive_words = {'growth', 'surge', 'rise', 'up', 'gain', 'bull', 'buy', 'strong', 'profit', 'beat', 'record', 'high', 'rally', 'positive', 'win', 'upgrade', 'dividend'}
        self.negative_words = {'drop', 'fall', 'down', 'loss', 'lose', 'miss', 'bear', 'sell', 'weak', 'low', 'crash', 'decline', 'negative', 'fail', 'debt', 'risk', 'downgrade', 'sue', 'cut'}

    def analyze(self, text):
        if not text: return 0.0
        words = re.findall(r'\w+', text.lower())
        if not words: return 0.0
        score = sum(1 for w in words if w in self.positive_words) - sum(1 for w in words if w in self.negative_words)
        return max(min(score / 5.0, 1.0), -1.0)

fallback_analyzer = BasicSentimentAnalyzer()

# --- Helper Functions ---
def get_symbol_from_name(query):
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        response = requests.get(url, params={'q': query, 'quotesCount': 1}, headers={'User-Agent': 'Mozilla/5.0'})
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes'][0]['symbol']
    except Exception:
        pass
    return None

def analyze_stock(symbol, market):
    try:
        # 1. Market Formatting
        if market == 'IN' and not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
            
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1y")
        
        if len(history) < 20:
            return {"symbol": symbol, "error": "Not enough historical data for analysis."}

        # 2. Price & SMA calculations
        latest_price = float(history['Close'].iloc[-1])
        history['SMA_20'] = history['Close'].rolling(window=20).mean()
        sma_20 = float(history['SMA_20'].iloc[-1])
        price_vs_sma_pct = (latest_price - sma_20) / sma_20
        
        # 3. Chart Data Extraction (Last 6 months)
        chart_dates = history.index.strftime('%Y-%m-%d').tolist()[-120:]
        chart_prices = history['Close'].round(2).tolist()[-120:]

        # 4. The Backtesting Engine (1-Year SMA Strategy)
        capital = 10000.0 
        shares = 0
        for i in range(20, len(history)):
            price = history['Close'].iloc[i]
            sma = history['SMA_20'].iloc[i]
            if price > sma and capital > price:
                shares_to_buy = capital // price
                shares += shares_to_buy
                capital -= (shares_to_buy * price)
            elif price < sma and shares > 0:
                capital += (shares * price)
                shares = 0
                
        final_value = capital + (shares * latest_price)
        backtest_return_pct = ((final_value - 10000.0) / 10000.0) * 100

        # 5. Hugging Face FinBERT Sentiment Analysis
        sentiment_score = 0
        sentiment_source = "None"
        news_titles = []
        
        try:
            news = ticker.news
            if news:
                # Grab the top 5 most recent headlines to keep API fast
                news_titles = [item.get('title', '') for item in news][:5]
        except Exception:
            pass

        if news_titles:
            # Try Hugging Face API First
            if HF_API_KEY:
                try:
                    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
                    payload = {"inputs": news_titles}
                    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=5)
                    
                    if response.status_code == 200:
                        hf_results = response.json()
                        total_score = 0
                        valid_count = 0
                        
                        # Parse FinBERT's output (positive, negative, neutral)
                        for result in hf_results:
                            if not result: continue
                            top_prediction = result[0] # Gets the highest confidence label
                            label = top_prediction.get('label', '')
                            score = top_prediction.get('score', 0)
                            
                            if label == 'positive':
                                total_score += score
                            elif label == 'negative':
                                total_score -= score
                            # neutral adds 0
                            valid_count += 1
                            
                        if valid_count > 0:
                            sentiment_score = total_score / valid_count
                            sentiment_source = "FinBERT AI (Hugging Face)"
                except Exception as e:
                    print(f"HF API Failed: {e}")

            # Fallback to local basic analyzer if HF failed or no key is present
            if sentiment_source == "None":
                total_score = sum(fallback_analyzer.analyze(title) for title in news_titles)
                sentiment_score = total_score / len(news_titles)
                sentiment_source = "Local Algorithmic Fallback"

        # 6. Decision Logic
        signal = "HOLD"
        rationale = "Mixed technical and sentiment indicators."
        
        # Adjust thresholds based on source accuracy
        bull_thresh = 0.2 if "FinBERT" in sentiment_source else 0.1
        bear_thresh = -0.2 if "FinBERT" in sentiment_source else -0.1
        
        if price_vs_sma_pct > 0 and sentiment_score > 0:
            signal = "STRONG BUY" if (price_vs_sma_pct > 0.02 and sentiment_score > bull_thresh) else "BUY"
            rationale = "Uptrend confirmed: Price above SMA with positive AI market sentiment."
        elif price_vs_sma_pct < 0 and sentiment_score < 0:
            signal = "STRONG SELL" if (price_vs_sma_pct < -0.02 and sentiment_score < bear_thresh) else "SELL"
            rationale = "Downtrend confirmed: Price below SMA with negative AI market sentiment."

        return {
            "symbol": symbol,
            "latest_price": round(latest_price, 2),
            "sma_20": round(sma_20, 2),
            "price_vs_sma_pct": round(price_vs_sma_pct * 100, 2),
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_source": sentiment_source,
            "signal": signal,
            "rationale": rationale,
            "backtest_return": round(backtest_return_pct, 2),
            "chart_dates": chart_dates,
            "chart_prices": chart_prices
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# --- Web Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    results = []
    if request.method == 'POST':
        market = request.form.get('market', 'US')
        user_input = request.form.get('symbols', '')
        raw_inputs = [s.strip() for s in user_input.split(',') if s.strip()]
        
        for query in raw_inputs:
            symbol = get_symbol_from_name(query) or query.upper()
            analysis = analyze_stock(symbol, market)
            results.append(analysis)
            
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
