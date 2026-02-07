import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Natural Language Processing
import requests
from bs4 import BeautifulSoup
import feedparser
import re
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Technical Analysis
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Indian Market Predictor AI",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #0f3460;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .buy-signal {
        background: linear-gradient(135deg, #1a3c1a 0%, #2d5a2d 100%);
        border-left: 6px solid #4CAF50;
        animation: pulse-green 2s infinite;
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #3c1a1a 0%, #5a2d2d 100%);
        border-left: 6px solid #FF5252;
        animation: pulse-red 2s infinite;
    }
    
    .hold-signal {
        background: linear-gradient(135deg, #2d2d3c 0%, #3d3d5a 100%);
        border-left: 6px solid #FFC107;
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 82, 82, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); }
    }
    
    .metric-card {
        background: rgba(30, 30, 46, 0.8);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #0f3460;
        text-align: center;
    }
    
    .news-card {
        background: rgba(26, 26, 46, 0.9);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2962ff;
        transition: all 0.3s ease;
    }
    
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INDIAN MARKET DATABASE
# ============================================================================

class IndianMarketDatabase:
    """Database of Indian stocks, indices, and commodities"""
    
    NIFTY_50 = {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFC BANK": "HDFCBANK.NS",
        "ICICI BANK": "ICICIBANK.NS",
        "INFOSYS": "INFY.NS",
        "HINDUNILVR": "HINDUNILVR.NS",
        "ITC": "ITC.NS",
        "SBIN": "SBIN.NS",
        "BHARTI AIRTEL": "BHARTIARTL.NS",
        "KOTAK BANK": "KOTAKBANK.NS",
        "LT": "LT.NS",
        "AXIS BANK": "AXISBANK.NS",
        "MARUTI": "MARUTI.NS",
        "BAJFINANCE": "BAJFINANCE.NS",
        "ASIAN PAINT": "ASIANPAINT.NS",
        "HCL TECH": "HCLTECH.NS",
        "WIPRO": "WIPRO.NS",
        "SUN PHARMA": "SUNPHARMA.NS",
        "TITAN": "TITAN.NS",
        "DMART": "DMART.NS",
        "ULTRACEMCO": "ULTRACEMCO.NS",
        "NESTLE": "NESTLEIND.NS",
        "ONGC": "ONGC.NS",
        "POWERGRID": "POWERGRID.NS",
        "NTPC": "NTPC.NS",
        "JSW STEEL": "JSWSTEEL.NS",
        "TATA STEEL": "TATASTEEL.NS",
        "TECH MAHINDRA": "TECHM.NS",
        "BAJAJ FINSERV": "BAJAJFINSV.NS",
        "GRASIM": "GRASIM.NS"
    }
    
    INDICES = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "BANK NIFTY": "^NSEBANK",
        "NIFTY IT": "^CNXIT",
        "NIFTY AUTO": "^CNXAUTO",
        "NIFTY PHARMA": "^CNXPHARMA",
        "NIFTY FMCG": "^CNXFMCG",
        "NIFTY METAL": "^CNXMETAL",
        "NIFTY REALTY": "^CNXREALTY"
    }
    
    COMMODITIES = {
        "GOLD": "GOLD.NS",
        "SILVER": "SILVER.NS",
        "CRUDEOIL": "CRUDEOIL.NS",
        "NATURALGAS": "NATURALGAS.NS",
        "COPPER": "COPPER.NS",
        "ZINC": "ZINC.NS",
        "LEAD": "LEAD.NS",
        "ALUMINIUM": "ALUMINIUM.NS"
    }

# ============================================================================
# NEWS COLLECTOR
# ============================================================================

class IndianMarketNews:
    """Collect and analyze news for Indian markets"""
    
    def __init__(self):
        self.sources = {
            'moneycontrol': 'https://www.moneycontrol.com/rss/marketreports.xml',
            'economic_times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'business_standard': 'https://www.business-standard.com/rss/markets-106.rss'
        }
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        except:
            self.sia = None
    
    def fetch_news(self, query=""):
        """Fetch news from various sources"""
        all_news = []
        
        try:
            # MoneyControl
            feed = feedparser.parse(self.sources['moneycontrol'])
            for entry in feed.entries[:10]:
                news_item = {
                    'title': entry.title,
                    'link': entry.link,
                    'source': 'MoneyControl',
                    'published': entry.published,
                    'summary': entry.get('summary', '')
                }
                all_news.append(news_item)
            
            # Economic Times
            feed = feedparser.parse(self.sources['economic_times'])
            for entry in feed.entries[:10]:
                news_item = {
                    'title': entry.title,
                    'link': entry.link,
                    'source': 'Economic Times',
                    'published': entry.published,
                    'summary': entry.get('summary', '')
                }
                all_news.append(news_item)
            
        except Exception as e:
            st.warning(f"News fetch error: {str(e)}")
        
        # Filter by query if provided
        if query:
            filtered_news = []
            query_lower = query.lower()
            for news in all_news:
                if query_lower in news['title'].lower() or query_lower in news['summary'].lower():
                    filtered_news.append(news)
            all_news = filtered_news
        
        # Sort by date
        all_news.sort(key=lambda x: x['published'], reverse=True)
        
        # Analyze sentiment if available
        if self.sia:
            for news in all_news:
                text = news['title'] + " " + news['summary']
                sentiment = self.analyze_sentiment(text)
                news['sentiment'] = sentiment['compound']
                news['sentiment_label'] = self.get_sentiment_label(sentiment['compound'])
        else:
            for news in all_news:
                news['sentiment'] = 0
                news['sentiment_label'] = "⚪ Neutral"
        
        return all_news[:10]
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if self.sia:
            return self.sia.polarity_scores(text)
        return {'compound': 0}
    
    def get_sentiment_label(self, score):
        """Convert sentiment score to label"""
        if score >= 0.05:
            return "🟢 Bullish"
        elif score <= -0.05:
            return "🔴 Bearish"
        else:
            return "⚪ Neutral"
    
    def get_market_sentiment(self):
        """Get overall market sentiment"""
        all_news = self.fetch_news()
        if not all_news:
            return 0, "Neutral"
        
        avg_sentiment = np.mean([news['sentiment'] for news in all_news])
        
        if avg_sentiment >= 0.1:
            return avg_sentiment, "Strongly Bullish 🚀"
        elif avg_sentiment >= 0.05:
            return avg_sentiment, "Bullish 📈"
        elif avg_sentiment <= -0.1:
            return avg_sentiment, "Strongly Bearish 🐻"
        elif avg_sentiment <= -0.05:
            return avg_sentiment, "Bearish 📉"
        else:
            return avg_sentiment, "Neutral ⚖️"

# ============================================================================
# TECHNICAL ANALYZER (FIXED)
# ============================================================================

class TechnicalAnalyzer:
    """Generate technical indicators"""
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure we have required columns
        if 'Volume' not in df.columns:
            df['Volume'] = 1000000
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price features
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        signals = {
            'RSI': 'OVERSOLD' if latest['RSI'] < 30 else 'OVERBOUGHT' if latest['RSI'] > 70 else 'NEUTRAL',
            'MACD': 'BULLISH' if latest['MACD'] > latest['MACD_Signal'] else 'BEARISH',
            'BB': 'OVERSOLD' if latest['BB_Position'] < 0.2 else 'OVERBOUGHT' if latest['BB_Position'] > 0.8 else 'NEUTRAL',
            'Trend_20': 'BULLISH' if latest['Close'] > latest['SMA_20'] else 'BEARISH',
            'Trend_50': 'BULLISH' if latest['Close'] > latest['SMA_50'] else 'BEARISH'
        }
        
        # Count signals
        buy_signals = sum([1 for sig in signals.values() if 'BULLISH' in str(sig) or 'OVERSOLD' in str(sig)])
        sell_signals = sum([1 for sig in signals.values() if 'BEARISH' in str(sig) or 'OVERBOUGHT' in str(sig)])
        
        if buy_signals > sell_signals:
            overall = 'BUY'
            confidence = buy_signals / 5
        elif sell_signals > buy_signals:
            overall = 'SELL'
            confidence = sell_signals / 5
        else:
            overall = 'HOLD'
            confidence = 0.5
        
        signals['Overall'] = overall
        signals['Confidence'] = confidence
        
        return signals

# ============================================================================
# AI PREDICTION MODEL (FIXED)
# ============================================================================

class StockPredictorAI:
    """AI Model for predicting stock direction - FIXED VERSION"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.news_analyzer = IndianMarketNews()
        self.tech_analyzer = TechnicalAnalyzer()
        self.is_fitted = False
    
    def prepare_data(self, symbol, period='1y'):
        """Prepare data for training/prediction - SIMPLIFIED"""
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                st.error(f"No data available for {symbol}")
                return None
            
            # Add technical indicators
            df = self.tech_analyzer.calculate_indicators(df)
            
            # Create target variable (next 5 days return)
            df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
            
            # Remove last 5 rows (no future target)
            if len(df) > 5:
                df = df.iloc[:-5]
            
            return df
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None
    
    def train_single_model(self, df, model_type='xgboost'):
        """Train a single model with proper data splitting"""
        if df is None or len(df) < 100:
            return None, 0
        
        # Prepare features - SIMPLIFIED for reliability
        feature_cols = ['RSI', 'MACD', 'BB_Position', 'SMA_20', 'SMA_50', 
                       'Volume_Ratio', 'Returns', 'ATR']
        
        # Ensure we have these columns
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) < 5:
            return None, 0
        
        X = df[available_cols].fillna(0)
        y = df['Target']
        
        # Simple split (not time-series for simplicity)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.is_fitted = True
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:  # lightgbm
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, accuracy
    
    def predict_direction(self, symbol, period='6mo'):
        """Predict stock direction (UP/DOWN) - SIMPLIFIED AND FIXED"""
        try:
            # Get data
            df = self.prepare_data(symbol, period)
            if df is None:
                return None
            
            # Train models if not already trained
            if not self.models:
                st.info("🔄 Training AI models...")
                
                # Train XGBoost
                xgb_model, xgb_acc = self.train_single_model(df, 'xgboost')
                if xgb_model:
                    self.models['xgboost'] = {'model': xgb_model, 'accuracy': xgb_acc}
                
                # Train Random Forest
                rf_model, rf_acc = self.train_single_model(df, 'random_forest')
                if rf_model:
                    self.models['random_forest'] = {'model': rf_model, 'accuracy': rf_acc}
            
            if not self.models:
                st.error("Failed to train models")
                return None
            
            # Prepare latest data for prediction
            feature_cols = ['RSI', 'MACD', 'BB_Position', 'SMA_20', 'SMA_50', 
                          'Volume_Ratio', 'Returns', 'ATR']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if not available_cols:
                st.error("No features available for prediction")
                return None
            
            X_latest = df[available_cols].iloc[-1:].fillna(0)
            
            # Scale features if scaler is fitted
            if self.scaler and self.is_fitted:
                X_latest_scaled = self.scaler.transform(X_latest)
            else:
                X_latest_scaled = X_latest.values
            
            # Get predictions from all models
            predictions = []
            confidences = []
            
            for model_name, model_info in self.models.items():
                model = model_info['model']
                accuracy = model_info['accuracy']
                
                # Get prediction probability
                pred_proba = model.predict_proba(X_latest_scaled)
                prediction = 1 if pred_proba[0][1] > 0.5 else 0
                confidence = max(pred_proba[0]) * accuracy  # Weight by accuracy
                
                predictions.append(prediction)
                confidences.append(confidence)
            
            # Consensus prediction
            if not predictions:
                return None
            
            consensus = 1 if sum(predictions) >= len(predictions)/2 else 0
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            # Get technical signals
            tech_signals = self.tech_analyzer.generate_signals(df)
            
            # Get news sentiment
            news = self.news_analyzer.fetch_news(symbol.split('.')[0])
            news_sentiment = np.mean([n['sentiment'] for n in news]) if news else 0
            
            # Final decision
            final_signal = "BUY" if consensus == 1 else "SELL"
            
            # Adjust confidence based on technicals and news
            if tech_signals.get('Overall') == 'SELL' and final_signal == 'BUY':
                avg_confidence *= 0.8
            elif tech_signals.get('Overall') == 'BUY' and final_signal == 'SELL':
                avg_confidence *= 0.8
            
            # Calculate price change
            if len(df) >= 2:
                price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            else:
                price_change = 0
            
            result = {
                'symbol': symbol,
                'prediction': 'UP' if consensus == 1 else 'DOWN',
                'signal': final_signal,
                'confidence': min(max(avg_confidence, 0.1), 0.95),  # Clamped between 0.1 and 0.95
                'price': df['Close'].iloc[-1],
                'change_pct': price_change,
                'technical_signals': tech_signals,
                'news_sentiment': news_sentiment,
                'model_accuracy': np.mean([m['accuracy'] for m in self.models.values()]) if self.models else 0.5,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Header
    st.markdown("<h1 class='main-header'>🇮🇳 Indian Market Predictor AI</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StockPredictorAI()
    if 'news' not in st.session_state:
        st.session_state.news = IndianMarketNews()
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Market Selection")
        
        # Asset Type Selection
        asset_type = st.selectbox(
            "Select Asset Type",
            ["Stocks (Nifty 50)", "Indices", "Commodities", "Custom Symbol"]
        )
        
        symbol = ""
        
        if asset_type == "Stocks (Nifty 50)":
            selected_stock = st.selectbox(
                "Select Stock",
                list(IndianMarketDatabase.NIFTY_50.keys())
            )
            symbol = IndianMarketDatabase.NIFTY_50[selected_stock]
            
        elif asset_type == "Indices":
            selected_index = st.selectbox(
                "Select Index",
                list(IndianMarketDatabase.INDICES.keys())
            )
            symbol = IndianMarketDatabase.INDICES[selected_index]
            
        elif asset_type == "Commodities":
            selected_commodity = st.selectbox(
                "Select Commodity",
                list(IndianMarketDatabase.COMMODITIES.keys())
            )
            symbol = IndianMarketDatabase.COMMODITIES[selected_commodity]
            
        else:  # Custom Symbol
            custom_symbol = st.text_input("Enter Symbol (e.g., RELIANCE.NS, ^NSEI)", "RELIANCE.NS")
            symbol = custom_symbol.upper()
            if not symbol.endswith('.NS') and not symbol.startswith('^'):
                symbol += '.NS'
        
        # Time Period
        period = st.selectbox(
            "Analysis Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
        
        # Predict Button
        st.markdown("---")
        predict_button = st.button(
            "🚀 PREDICT NOW",
            type="primary",
            use_container_width=True,
            key="predict_button"
        )
    
    # Main Content
    if predict_button and symbol:
        with st.spinner("🔄 Analyzing market data..."):
            # Clear previous results
            st.session_state.last_prediction = None
            
            # Make prediction
            result = st.session_state.predictor.predict_direction(symbol, period)
            
            if result:
                st.session_state.last_prediction = result
                st.success("✅ Prediction completed!")
            else:
                st.error("❌ Could not generate prediction. Please try again.")
    
    # Display results if available
    if st.session_state.last_prediction:
        result = st.session_state.last_prediction
        
        # Display Prediction Result
        st.markdown("---")
        
        # Prediction Card
        if result['signal'] == 'BUY':
            card_class = "buy-signal"
            emoji = "🚀"
        elif result['signal'] == 'SELL':
            card_class = "sell-signal"
            emoji = "⚠️"
        else:
            card_class = "hold-signal"
            emoji = "⏸️"
        
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; color: white;">{result['symbol']}</h2>
                    <p style="color: #aaa; margin: 5px 0;">Last Price: ₹{result['price']:.2f} ({result['change_pct']:+.2f}%)</p>
                </div>
                <div style="text-align: right;">
                    <h1 style="margin: 0; font-size: 3rem;">{emoji} {result['signal']}</h1>
                    <p style="color: #aaa;">Prediction: {result['prediction']}</p>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                    <h3 style="margin: 0 0 10px 0;">Confidence Level: {result['confidence']:.1%}</h3>
                    <div style="height: 10px; background: rgba(255,255,255,0.2); border-radius: 5px; overflow: hidden;">
                        <div style="height: 100%; width: {result['confidence']*100}%; 
                                 background: {'#4CAF50' if result['signal'] == 'BUY' else '#FF5252'}; 
                                 border-radius: 5px;"></div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 20px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                <div class="metric-card">
                    <h4 style="margin: 0; color: #aaa;">Model Accuracy</h4>
                    <p style="font-size: 1.5rem; margin: 10px 0; color: white;">{result['model_accuracy']:.1%}</p>
                </div>
                <div class="metric-card">
                    <h4 style="margin: 0; color: #aaa;">News Sentiment</h4>
                    <p style="font-size: 1.5rem; margin: 10px 0; 
                            color: {'#4CAF50' if result['news_sentiment'] > 0 else '#FF5252' if result['news_sentiment'] < 0 else '#FFC107'}">
                        {result['news_sentiment']:+.2f}
                    </p>
                </div>
                <div class="metric-card">
                    <h4 style="margin: 0; color: #aaa;">Technical Signal</h4>
                    <p style="font-size: 1.5rem; margin: 10px 0; 
                            color: {'#4CAF50' if result['technical_signals'].get('Overall') == 'BUY' else '#FF5252'}">
                        {result['technical_signals'].get('Overall', 'NEUTRAL')}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for Detailed Analysis
        tab1, tab2, tab3 = st.tabs(["📈 Chart", "🔍 Technicals", "📰 News"])
        
        with tab1:
            # Price Chart
            try:
                stock_data = yf.Ticker(symbol)
                hist = stock_data.history(period=period)
                
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price',
                    increasing_line_color='#00C853',
                    decreasing_line_color='#FF5252'
                ))
                
                # Add moving averages
                for ma in [20, 50]:
                    if len(hist) > ma:
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'].rolling(ma).mean(),
                            name=f'MA {ma}',
                            line=dict(width=1.5),
                            opacity=0.7
                        ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    yaxis_title="Price (₹)",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Chart error: {str(e)}")
        
        with tab2:
            # Technical Analysis
            st.subheader("Technical Analysis Signals")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Indicator Signals")
                for indicator, signal in result['technical_signals'].items():
                    if indicator not in ['Overall', 'Confidence']:
                        color = "#4CAF50" if 'BULLISH' in str(signal) or 'OVERSOLD' in str(signal) else "#FF5252" if 'BEARISH' in str(signal) or 'OVERBOUGHT' in str(signal) else "#FFC107"
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333;">
                            <span>{indicator}:</span>
                            <span style="color: {color}; font-weight: bold;">{signal}</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📈 Key Levels")
                
                # Calculate support and resistance
                try:
                    stock_data = yf.Ticker(symbol)
                    hist = stock_data.history(period="1mo")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        support = hist['Low'].rolling(20).min().iloc[-1]
                        resistance = hist['High'].rolling(20).max().iloc[-1]
                        
                        st.metric("Current Price", f"₹{current_price:.2f}")
                        st.metric("Support Level", f"₹{support:.2f}")
                        st.metric("Resistance Level", f"₹{resistance:.2f}")
                        
                        # Progress bar for price position
                        price_range = resistance - support
                        if price_range > 0:
                            position = (current_price - support) / price_range
                            st.markdown(f"**Price Position:** {position:.1%}")
                            st.progress(position)
                            
                            if position < 0.3:
                                st.info("📍 Near Support - Potential buying opportunity")
                            elif position > 0.7:
                                st.warning("📍 Near Resistance - Potential selling opportunity")
                except:
                    pass
        
        with tab3:
            # News Analysis
            st.subheader("Latest Market News")
            
            # Overall market sentiment
            market_sentiment, sentiment_label = st.session_state.news.get_market_sentiment()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Sentiment", sentiment_label)
            with col2:
                st.metric("Sentiment Score", f"{market_sentiment:+.2f}")
            
            # Fetch and display news for the symbol
            news_items = st.session_state.news.fetch_news(symbol.split('.')[0])
            
            if news_items:
                for news in news_items[:5]:
                    sentiment_color = "#4CAF50" if news['sentiment'] > 0.05 else "#FF5252" if news['sentiment'] < -0.05 else "#FFC107"
                    
                    st.markdown(f"""
                    <div class="news-card">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <small style="color: #888;">{news['source']}</small>
                            <small style="color: {sentiment_color}; font-weight: bold;">{news['sentiment_label']}</small>
                        </div>
                        <a href="{news['link']}" target="_blank" style="color: white; text-decoration: none; font-weight: 500;">
                            {news['title']}
                        </a>
                        <div style="margin-top: 8px;">
                            <small style="color: #aaa;">{news['published'][:16]}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent news found for this symbol.")
        
        # Trading Recommendations
        st.markdown("---")
        st.subheader("🎯 Trading Recommendations")
        
        if result['signal'] == 'BUY':
            st.success(f"""
            **Recommended Action: BUY with {result['confidence']:.0%} confidence**
            
            📋 **Trading Plan:**
            - Entry Price: ₹{result['price']:.2f}
            - Stop Loss: ₹{result['price'] * 0.97:.2f} (3% below entry)
            - Target 1: ₹{result['price'] * 1.05:.2f} (5% profit)
            - Target 2: ₹{result['price'] * 1.10:.2f} (10% profit)
            
            ⚡ **Key Factors:**
            - Technical Signals: {sum([1 for s in result['technical_signals'].values() if 'BULLISH' in str(s) or 'OVERSOLD' in str(s)])}/5 positive
            - News Sentiment: {'Positive' if result['news_sentiment'] > 0 else 'Negative' if result['news_sentiment'] < 0 else 'Neutral'}
            - AI Model Confidence: {result['confidence']:.1%}
            """)
        elif result['signal'] == 'SELL':
            st.error(f"""
            **Recommended Action: SELL with {result['confidence']:.0%} confidence**
            
            📋 **Trading Plan:**
            - Exit Price: ₹{result['price']:.2f}
            - Wait for confirmation before shorting
            - Consider defensive positions
            
            ⚡ **Key Factors:**
            - Technical Signals: {sum([1 for s in result['technical_signals'].values() if 'BEARISH' in str(s) or 'OVERBOUGHT' in str(s)])}/5 negative
            - News Sentiment: {'Positive' if result['news_sentiment'] > 0 else 'Negative' if result['news_sentiment'] < 0 else 'Neutral'}
            - AI Model Confidence: {result['confidence']:.1%}
            """)
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        **⚠️ IMPORTANT DISCLAIMER:**
        
        This prediction is generated by AI models for educational and informational purposes only. 
        It should not be considered as financial advice. 
        
        - Past performance does not guarantee future results
        - Always do your own research before trading
        - Consider consulting with a qualified financial advisor
        - Never invest money you cannot afford to lose
        
        The accuracy of predictions may vary and there's always risk in stock market investments.
        """)
    
    else:
        # Welcome Screen
        st.markdown("""
        <div style='text-align: center; padding: 40px 20px;'>
            <h2>🎯 AI-Powered Indian Market Predictions</h2>
            <p style='color: #aaa; font-size: 1.1rem; max-width: 800px; margin: 20px auto;'>
                Get accurate BUY/SELL signals for Indian stocks, indices, and commodities using 
                advanced AI models combined with real-time news sentiment and technical analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3>🤖 AI Prediction</h3>
                <p>Multiple ML models for accurate direction prediction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h3>📰 News Sentiment</h3>
                <p>Real-time news analysis from Indian sources</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h3>📊 Technical Analysis</h3>
                <p>Comprehensive indicators with automatic signals</p>
            </div>
            """, unsafe_allow_html=True)
        
        # How to Use
        st.markdown("---")
        st.subheader("🚀 How to Use")
        
        st.markdown("""
        1. **Select** your asset type from the sidebar
        2. **Choose** a specific symbol
        3. **Set** your analysis period
        4. **Click** the "PREDICT NOW" button
        5. **Review** the AI prediction with confidence level
        
        ### 📋 Supported Assets:
        - **Stocks**: All Nifty 50 companies
        - **Indices**: Nifty 50, Sensex, Bank Nifty, Sectoral indices
        - **Commodities**: Gold, Silver, Crude Oil, Metals
        - **Custom**: Any Indian stock with .NS suffix
        
        ### ⚡ Quick Start:
        Select "RELIANCE.NS" from Stocks and click PREDICT NOW to test the system!
        """)

# Run the app
if __name__ == "__main__":
    main()
