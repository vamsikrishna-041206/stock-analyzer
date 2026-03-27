from flask import Flask, request, render_template
import yfinance as yf
import requests
import re
from datetime import datetime, timezone, timedelta
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from supabase import create_client, Client
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

app = Flask(__name__)

# --- Configuration & Keys ---
HF_API_KEY = os.environ.get('HF_API_KEY')
HF_API_URL = "https://api.huggingface.co/models/ProsusAI/finbert"

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

# --- Initialize Database ---
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase Init Error: {e}")

# --- Fallback Sentiment Analyzer ---
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

def get_google_news_headlines(symbol, market):
    """Fetches the latest headlines from Google News to bypass Yahoo's missing data."""
    try:
        clean_symbol = symbol.replace('.NS', '')
        query = urllib.parse.quote(f"{clean_symbol} stock finance news")
        
        # Adjust Google News region based on selected market
        if market == 'IN':
            url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        else:
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        titles = []
        for item in root.findall('.//item')[:5]: # Grab the top 5 most recent headlines
            title = item.find('title')
            if title is not None and title.text:
                titles.append(title.text)
        return titles
    except Exception as e:
        print(f"Google News Fetch Error: {e}")
        return []

# --- Core Algorithmic Engine ---
def analyze_stock(symbol, market):
    try:
        # 1. Market Formatting (Indian vs US)
        if market == 'IN' and not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
            
        # 2. Database Cache Check (Zero-Latency Load)
        if supabase:
            try:
                cache_res = supabase.table('stock_cache').select('*').eq('symbol', symbol).execute()
                if len(cache_res.data) > 0:
                    cached_row = cache_res.data[0]
                    last_updated_str = cached_row['last_updated']
                    last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
                    
                    # Serve from database if less than 1 hour old
                    if datetime.now(timezone.utc) - last_updated < timedelta(hours=1):
                        print(f"✅ Served {symbol} instantly from Supabase Cache!")
                        return cached_row['data']
            except Exception as e:
                print(f"Cache Read Error: {e}")

        # 3. Fetch Fresh Data from Yahoo Finance
        print(f"⏳ Fetching fresh data for {symbol}...")
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1y")
        
        if len(history) < 20:
            return {"symbol": symbol, "error": "Not enough historical data for analysis."}

        # 4. Price & SMA calculations
        latest_price = float(history['Close'].iloc[-1])
        history['SMA_20'] = history['Close'].rolling(window=20).mean()
        sma_20 = float(history['SMA_20'].iloc[-1])
        price_vs_sma_pct = (latest_price - sma_20) / sma_20
        
        # 5. Chart Data & Predictive Machine Learning (Scikit-Learn)
        recent_history = history.tail(120).copy()
        chart_dates = recent_history.index.strftime('%Y-%m-%d').tolist()
        chart_prices = recent_history['Close'].round(2).tolist()

        # Prepare data for AI Linear Regression
        recent_history['Days_From_Start'] = np.arange(len(recent_history))
        X = recent_history[['Days_From_Start']]
        y = recent_history['Close']

        # Train Model
        model = LinearRegression()
        model.fit(X, y)

        # Predict next 7 days
        last_day_index = len(recent_history)
        future_X = np.arange(last_day_index, last_day_index + 7).reshape(-1, 1)
        future_predictions = model.predict(future_X)

        # Generate future dates and prices
        last_date = recent_history.index[-1]
        future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
        future_prices = [round(p, 2) for p in future_predictions]

        # 6. The Backtesting Engine (1-Year SMA Strategy Simulation)
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

        # 7. Sentiment Analysis (Google News + FinBERT)
        sentiment_score = 0
        sentiment_source = "None"
        
        news_titles = get_google_news_headlines(symbol, market)
        print(f"📰 Found {len(news_titles)} headlines for {symbol}")

        if news_titles:
            # Attempt AI inference via Hugging Face API
            if HF_API_KEY:
                try:
                    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
                    payload = {"inputs": news_titles}
                    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=5)
                    
                    if response.status_code == 200:
                        hf_results = response.json()
                        total_score = 0
                        valid_count = 0
                        
                        for result in hf_results:
                            if not result: continue
                            top_prediction = result[0] 
                            label = top_prediction.get('label', '')
                            score = top_prediction.get('score', 0)
                            
                            if label == 'positive': total_score += score
                            elif label == 'negative': total_score -= score
                            valid_count += 1
                            
                        if valid_count > 0:
                            sentiment_score = total_score / valid_count
                            sentiment_source = "FinBERT AI (Hugging Face)"
                except Exception as e:
                    print(f"HF API Failed: {e}")

            # Fallback to local dictionary if HF fails
            if sentiment_source == "None":
                total_score = sum(fallback_analyzer.analyze(title) for title in news_titles)
                sentiment_score = total_score / len(news_titles)
                sentiment_source = "Local Algorithmic Fallback"

        # 8. Core Decision Logic
        signal = "HOLD"
        rationale = "Mixed technical and sentiment indicators."
        bull_thresh = 0.2 if "FinBERT" in sentiment_source else 0.1
        bear_thresh = -0.2 if "FinBERT" in sentiment_source else -0.1
        
        if price_vs_sma_pct > 0 and sentiment_score > 0:
            signal = "STRONG BUY" if (price_vs_sma_pct > 0.02 and sentiment_score > bull_thresh) else "BUY"
            rationale = "Uptrend confirmed: Price above SMA with positive market sentiment."
        elif price_vs_sma_pct < 0 and sentiment_score < 0:
            signal = "STRONG SELL" if (price_vs_sma_pct < -0.02 and sentiment_score < bear_thresh) else "SELL"
            rationale = "Downtrend confirmed: Price below SMA with negative market sentiment."

        # Compile final dictionary to send to frontend
        final_result = {
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
            "chart_prices": chart_prices,
            "future_dates": future_dates,
            "future_prices": future_prices
        }

        # 9. Save to Database Cache
        if supabase:
            try:
                supabase.table('stock_cache').upsert({
                    'symbol': symbol, 
                    'data': final_result,
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }).execute()
                print(f"💾 Saved {symbol} to Supabase Cache.")
            except Exception as e:
                print(f"Cache Write Error: {e}")

        return final_result

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
