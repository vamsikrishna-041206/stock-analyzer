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
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, params={'q': query, 'quotesCount': 1}, headers=headers)
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes'][0]['symbol']
    except Exception:
        pass
    return None

def get_google_news_headlines(symbol, market):
    try:
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
        query = urllib.parse.quote(f"{clean_symbol} stock finance news")
        
        if market == 'IN':
            url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        else:
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        titles = []
        for item in root.findall('.//item')[:5]:
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
        # 1. THE EXCHANGE HUNTER
        if market == 'IN':
            base_symbol = symbol.replace('.NS', '').replace('.BO', '')
            suffixes = ['.NS', '.BO'] 
        else:
            base_symbol = symbol
            suffixes = ['']

        history = None
        valid_symbol = symbol

        for suffix in suffixes:
            test_symbol = f"{base_symbol}{suffix}"
            
            # Database Cache Check
            if supabase:
                try:
                    cache_res = supabase.table('stock_cache').select('*').eq('symbol', test_symbol).execute()
                    if len(cache_res.data) > 0:
                        cached_row = cache_res.data[0]
                        last_updated_str = cached_row['last_updated']
                        last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
                        
                        if datetime.now(timezone.utc) - last_updated < timedelta(hours=1):
                            print(f"✅ Served {test_symbol} instantly from Supabase Cache!")
                            return cached_row['data']
                except Exception as e:
                    pass

            # Fetch Fresh Data (Letting yfinance handle its own disguise)
            print(f"⏳ Fetching fresh data for {test_symbol}...")
            ticker = yf.Ticker(test_symbol) # Removed our custom session
            temp_history = ticker.history(period="1y")
            
            # Verify the dataframe isn't empty and has enough data
            if temp_history is not None and not temp_history.empty and len(temp_history) >= 20:
                history = temp_history
                valid_symbol = test_symbol
                break

        # Safety Net: If Yahoo completely blocked us or the stock doesn't exist
        if history is None or history.empty or len(history) < 20:
            return {"symbol": symbol, "error": "ERR_NO_DATA: Ensure you use the exact ticker symbol (e.g. RELIANCE, AAPL) or Yahoo servers are temporarily blocking the connection."}

        symbol = valid_symbol

        # 4. Price & SMA calculations
        latest_price = float(history['Close'].iloc[-1])
        history['SMA_20'] = history['Close'].rolling(window=20).mean()
        sma_20 = float(history['SMA_20'].iloc[-1])
        price_vs_sma_pct = (latest_price - sma_20) / sma_20
        
        # 5. Chart Data & AI Projection
        recent_history = history.tail(120).copy()
        chart_dates = recent_history.index.strftime('%Y-%m-%d').tolist()
        chart_prices = recent_history['Close'].round(2).tolist()

        recent_history['Days_From_Start'] = np.arange(len(recent_history))
        X = recent_history[['Days_From_Start']]
        y = recent_history['Close']

        model = LinearRegression()
        model.fit(X, y)

        last_day_index = len(recent_history)
        future_X = np.arange(last_day_index, last_day_index + 7).reshape(-1, 1)
        future_predictions = model.predict(future_X)

        last_date = recent_history.index[-1]
        future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
        future_prices = [round(p, 2) for p in future_predictions]

        # 6. Backtesting Engine
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

        # 7. Sentiment Analysis
        sentiment_score = 0
        sentiment_source = "None"
        
        news_titles = get_google_news_headlines(symbol, market)

        if news_titles:
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
                            sentiment_source = "FinBERT AI"
                except Exception as e:
                    pass

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

        # 9. Generate Executive Summary
        trend_direction = "above" if price_vs_sma_pct > 0 else "below"
        sentiment_tone = "highly positive" if sentiment_score > 0.15 else "mildly positive" if sentiment_score > 0 else "highly negative" if sentiment_score < -0.15 else "mildly negative" if sentiment_score < 0 else "neutral"

        detailed_summary = (
            f"DESCRIPTION: {symbol} is currently trading at ${latest_price:.2f}, which is {abs(price_vs_sma_pct * 100):.2f}% {trend_direction} its 20-day Simple Moving Average. "
            f"Simultaneously, the qualitative AI analysis via {sentiment_source} reflects a {sentiment_tone} market sentiment (Index: {sentiment_score:.2f}).\n\n"
            f"CONCLUSION: The 1-year algorithmic backtest of this specific asset yielded a {backtest_return_pct:.2f}% return. "
            f"Fusing the technical trend momentum with the current AI sentiment projections, the system firmly concludes with a {signal} directive."
        )

        final_result = {
            "symbol": symbol,
            "latest_price": round(latest_price, 2),
            "sma_20": round(sma_20, 2),
            "price_vs_sma_pct": round(price_vs_sma_pct * 100, 2),
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_source": sentiment_source,
            "signal": signal,
            "rationale": rationale,
            "summary": detailed_summary,
            "backtest_return": round(backtest_return_pct, 2),
            "chart_dates": chart_dates,
            "chart_prices": chart_prices,
            "future_dates": future_dates,
            "future_prices": future_prices
        }

        # 10. Save to Database Cache
        if supabase:
            try:
                supabase.table('stock_cache').upsert({
                    'symbol': symbol, 
                    'data': final_result,
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }).execute()
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
            if market == 'IN':
                symbol = query.upper().replace(' ', '')
            else:
                symbol = get_symbol_from_name(query) or query.upper()
                
            analysis = analyze_stock(symbol, market)
            results.append(analysis)
            
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
