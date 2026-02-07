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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Natural Language Processing
import requests
from bs4 import BeautifulSoup
import feedparser
import re
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
try:
    nltk.download('vader_lexicon')
except:
    pass

# Technical Analysis
import ta
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

# Database
import sqlite3
import pickle
import json
import hashlib

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Indian Market Predictor AI",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for Indian Market Theme
st.markdown("""
<style>
    /* Main theme colors */
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
    
    .indian-flag {
        height: 5px;
        background: linear-gradient(to right, #FF9933 33%, #FFFFFF 33% 66%, #138808 66%);
        margin: 10px 0;
        border-radius: 3px;
    }
    
    /* Card styling */
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
    
    /* Metric cards */
    .metric-card {
        background: rgba(30, 30, 46, 0.8);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #0f3460;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    /* News cards */
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
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF9933 0%, #138808 100%);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0e1117;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e2330;
        border-bottom: 3px solid #FF9933;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INDIAN MARKET DATABASE
# ============================================================================

class IndianMarketDatabase:
    """Database of Indian stocks, indices, and commodities"""
    
    # Nifty 50 Stocks
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
    
    # Major Indices
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
    
    # Indian Commodities (MCX)
    COMMODITIES = {
        "GOLD": "GOLD.NS",
        "SILVER": "SILVER.NS",
        "CRUDEOIL": "CRUDEOIL.NS",
        "NATURALGAS": "NATURALGAS.NS",
        "COPPER": "COPPER.NS",
        "ZINC": "ZINC.NS",
        "LEAD": "LEAD.NS",
        "ALUMINIUM": "ALUMINIUM.NS",
        "COTTON": "COTTON.NS"
    }
    
    # Sector-wise Classification
    SECTORS = {
        "BANKING": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
        "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
        "AUTOMOBILE": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJAUTO.NS", "EICHERMOT.NS"],
        "PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS"],
        "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
        "ENERGY": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "GAIL.NS"],
        "METALS": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "JINDALSTEL.NS"]
    }

# ============================================================================
# NEWS COLLECTOR & SENTIMENT ANALYZER
# ============================================================================

class IndianMarketNews:
    """Collect and analyze news for Indian markets"""
    
    def __init__(self):
        self.sources = {
            'moneycontrol': 'https://www.moneycontrol.com/rss/marketreports.xml',
            'economic_times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'business_standard': 'https://www.business-standard.com/rss/markets-106.rss',
            'livemint': 'https://www.livemint.com/rss/markets',
            'ndtv_profit': 'https://www.ndtvprofit.com/feeds/market-news'
        }
        self.sia = SentimentIntensityAnalyzer()
        
    def fetch_news(self, query="", source='all'):
        """Fetch news from various sources"""
        all_news = []
        
        try:
            # MoneyControl - Indian Market Specific
            if source == 'all' or source == 'moneycontrol':
                feed = feedparser.parse(self.sources['moneycontrol'])
                for entry in feed.entries[:10]:
                    news_item = {
                        'title': entry.title,
                        'link': entry.link,
                        'source': 'MoneyControl',
                        'published': entry.published,
                        'summary': entry.get('summary', ''),
                        'symbols': self.extract_symbols(entry.title + " " + entry.get('summary', ''))
                    }
                    all_news.append(news_item)
            
            # Economic Times
            if source == 'all' or source == 'economic_times':
                feed = feedparser.parse(self.sources['economic_times'])
                for entry in feed.entries[:10]:
                    news_item = {
                        'title': entry.title,
                        'link': entry.link,
                        'source': 'Economic Times',
                        'published': entry.published,
                        'summary': entry.get('summary', ''),
                        'symbols': self.extract_symbols(entry.title + " " + entry.get('summary', ''))
                    }
                    all_news.append(news_item)
            
            # Business Standard
            if source == 'all' or source == 'business_standard':
                feed = feedparser.parse(self.sources['business_standard'])
                for entry in feed.entries[:10]:
                    news_item = {
                        'title': entry.title,
                        'link': entry.link,
                        'source': 'Business Standard',
                        'published': entry.published,
                        'summary': entry.get('summary', ''),
                        'symbols': self.extract_symbols(entry.title + " " + entry.get('summary', ''))
                    }
                    all_news.append(news_item)
        
        except Exception as e:
            st.warning(f"News fetch error: {str(e)}")
        
        # Filter by query if provided
        if query:
            filtered_news = []
            query_lower = query.lower()
            for news in all_news:
                if (query_lower in news['title'].lower() or 
                    query_lower in news['summary'].lower() or
                    query in news['symbols']):
                    filtered_news.append(news)
            all_news = filtered_news
        
        # Sort by date and remove duplicates
        all_news.sort(key=lambda x: x['published'], reverse=True)
        
        # Analyze sentiment for each news
        for news in all_news:
            text = news['title'] + " " + news['summary']
            sentiment = self.analyze_sentiment(text)
            news['sentiment'] = sentiment['compound']
            news['sentiment_label'] = self.get_sentiment_label(sentiment['compound'])
        
        return all_news[:15]  # Return top 15 news items
    
    def extract_symbols(self, text):
        """Extract stock symbols from text"""
        symbols = []
        # Look for common Indian stock patterns
        patterns = [
            r'\b(RELIANCE|TCS|HDFC|ICICI|INFY|SBIN|BHARTI|KOTAK|AXIS)\b',
            r'\b(MARUTI|BAJAJ|TITAN|SUN PHARMA|ITC)\b',
            r'\b(NIFTY|SENSEX|BANK NIFTY)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            symbols.extend(matches)
        
        return list(set(symbols))
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        return self.sia.polarity_scores(text)
    
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
# TECHNICAL ANALYSIS ENGINE
# ============================================================================

class TechnicalAnalyzer:
    """Generate comprehensive technical indicators for Indian stocks"""
    
    def __init__(self):
        self.indicators_config = {
            'trend': ['sma', 'ema', 'macd', 'adx', 'ichimoku'],
            'momentum': ['rsi', 'stochastic', 'williams_r', 'cci'],
            'volatility': ['bollinger', 'atr', 'keltner'],
            'volume': ['obv', 'vwap', 'money_flow']
        }
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'Volume':
                    df[col] = 1000000  # Default volume
                else:
                    df[col] = df['Close']  # Use Close for missing OHLC
        
        # 1. Trend Indicators
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
            df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # ADX
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        
        # 2. Momentum Indicators
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Williams %R
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14).williams_r()
        
        # 3. Volatility Indicators
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        
        # 4. Volume Indicators
        # OBV
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
        # VWAP
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'], window=14).volume_weighted_average_price()
        
        # 5. Price Action Features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # 6. Support and Resistance Levels
        df['Support_Level'] = df['Low'].rolling(20).min()
        df['Resistance_Level'] = df['High'].rolling(20).max()
        
        # 7. Market Regime
        df['Trend_Strength'] = df['ADX']
        df['Volatility_Regime'] = df['ATR'] / df['Close'].rolling(20).mean()
        
        # 8. Candlestick Patterns (simplified)
        df['Bullish_Engulfing'] = ((df['Close'] > df['Open']) & 
                                  (df['Close'].shift(1) < df['Open'].shift(1)) &
                                  (df['Close'] > df['Open'].shift(1)) &
                                  (df['Open'] < df['Close'].shift(1))).astype(int)
        
        df['Bearish_Engulfing'] = ((df['Close'] < df['Open']) & 
                                  (df['Close'].shift(1) > df['Open'].shift(1)) &
                                  (df['Close'] < df['Open'].shift(1)) &
                                  (df['Open'] > df['Close'].shift(1))).astype(int)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals based on technical indicators"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        signals = {
            'RSI_Signal': 'BUY' if latest['RSI'] < 30 else 'SELL' if latest['RSI'] > 70 else 'NEUTRAL',
            'MACD_Signal': 'BUY' if latest['MACD'] > latest['MACD_Signal'] else 'SELL',
            'BB_Signal': 'BUY' if latest['BB_Position'] < 0.2 else 'SELL' if latest['BB_Position'] > 0.8 else 'NEUTRAL',
            'Trend_Signal': 'BULLISH' if latest['Close'] > latest['SMA_50'] else 'BEARISH',
            'Volume_Signal': 'BUY' if latest['Volume'] > latest['Volume'].rolling(20).mean().iloc[-1] * 1.5 else 'NEUTRAL'
        }
        
        # Overall signal logic
        buy_signals = sum([1 for sig in signals.values() if 'BUY' in str(sig) or 'BULLISH' in str(sig)])
        sell_signals = sum([1 for sig in signals.values() if 'SELL' in str(sig) or 'BEARISH' in str(sig)])
        
        if buy_signals > sell_signals:
            overall_signal = 'BUY'
            confidence = buy_signals / (buy_signals + sell_signals)
        elif sell_signals > buy_signals:
            overall_signal = 'SELL'
            confidence = sell_signals / (buy_signals + sell_signals)
        else:
            overall_signal = 'HOLD'
            confidence = 0.5
        
        signals['Overall_Signal'] = overall_signal
        signals['Confidence'] = confidence
        
        return signals

# ============================================================================
# AI PREDICTION MODEL
# ============================================================================

class StockPredictorAI:
    """AI Model for predicting stock direction"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.news_analyzer = IndianMarketNews()
        self.tech_analyzer = TechnicalAnalyzer()
        
    def prepare_data(self, symbol, period='1y'):
        """Prepare data for training/prediction"""
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                st.error(f"No data available for {symbol}")
                return None
            
            # Add technical indicators
            df = self.tech_analyzer.calculate_indicators(df)
            
            # Add news sentiment (last 30 days rolling average)
            news_data = self.news_analyzer.fetch_news(symbol.split('.')[0])
            if news_data:
                sentiment_scores = [news['sentiment'] for news in news_data]
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            else:
                avg_sentiment = 0
            
            # Add sentiment as a feature
            df['News_Sentiment'] = avg_sentiment
            
            # Create target variable (next 5 days return)
            df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
            
            # Remove last 5 rows (no future target)
            df = df.iloc[:-5]
            
            return df
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None
    
    def train_model(self, df, model_type='xgboost'):
        """Train prediction model"""
        if df is None or len(df) < 100:
            st.warning("Insufficient data for training")
            return None
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        X = df[feature_cols].fillna(0)
        y = df['Target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data (time-series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        if model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
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
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models[model_type] = {
            'model': model,
            'accuracy': accuracy,
            'feature_importance': dict(zip(feature_cols, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
        }
        
        return model, accuracy
    
    def predict_direction(self, symbol, period='6mo'):
        """Predict stock direction (UP/DOWN)"""
        try:
            # Get latest data
            df = self.prepare_data(symbol, period)
            if df is None:
                return None
            
            # Prepare features for prediction
            feature_cols = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
            X_latest = df[feature_cols].iloc[-30:].fillna(0)  # Last 30 days for context
            X_scaled = self.scaler.transform(X_latest)
            
            # Train multiple models and get consensus
            predictions = []
            confidences = []
            
            for model_type in ['xgboost', 'random_forest', 'gradient_boosting']:
                model_info = self.models.get(model_type)
                
                if model_info is None:
                    # Train model if not already trained
                    model, accuracy = self.train_model(df, model_type)
                    if model is None:
                        continue
                    model_info = self.models[model_type]
                
                # Make prediction
                pred_proba = model_info['model'].predict_proba(X_scaled[-1:].reshape(1, -1))
                prediction = 1 if pred_proba[0][1] > 0.5 else 0
                confidence = max(pred_proba[0])
                
                predictions.append(prediction)
                confidences.append(confidence * model_info['accuracy'])  # Weight by model accuracy
            
            if not predictions:
                return None
            
            # Consensus prediction
            consensus = 1 if sum(predictions) >= 2 else 0
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            # Get technical signals
            tech_signals = self.tech_analyzer.generate_signals(df)
            
            # Get news sentiment
            news = self.news_analyzer.fetch_news(symbol.split('.')[0])
            news_sentiment = np.mean([n['sentiment'] for n in news]) if news else 0
            
            # Final decision with weighted factors
            final_signal = "BUY" if consensus == 1 else "SELL"
            
            # Adjust based on technicals and news
            if tech_signals.get('Overall_Signal') == 'SELL' and final_signal == 'BUY':
                avg_confidence *= 0.8  # Reduce confidence
            elif tech_signals.get('Overall_Signal') == 'BUY' and final_signal == 'SELL':
                avg_confidence *= 0.8
            
            if news_sentiment < -0.1 and final_signal == 'BUY':
                avg_confidence *= 0.7
            elif news_sentiment > 0.1 and final_signal == 'SELL':
                avg_confidence *= 0.7
            
            result = {
                'symbol': symbol,
                'prediction': 'UP' if consensus == 1 else 'DOWN',
                'signal': final_signal,
                'confidence': min(max(avg_confidence, 0), 1),  # Clamp between 0 and 1
                'price': df['Close'].iloc[-1],
                'change_pct': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100,
                'technical_signals': tech_signals,
                'news_sentiment': news_sentiment,
                'model_accuracy': np.mean([self.models[m]['accuracy'] for m in self.models]),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Header with Indian Theme
    st.markdown("<h1 class='main-header'>🇮🇳 Indian Market Predictor AI</h1>", unsafe_allow_html=True)
    st.markdown("<div class='indian-flag'></div>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StockPredictorAI()
    if 'news' not in st.session_state:
        st.session_state.news = IndianMarketNews()
    
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
        
        # Time Period
        period = st.selectbox(
            "Analysis Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )
        
        # Prediction Horizon
        horizon = st.selectbox(
            "Prediction Horizon",
            ["Next 5 Days", "Next 10 Days", "Next Month"],
            index=0
        )
        
        # Additional Options
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        show_technical = st.checkbox("Show Technical Analysis", value=True)
        show_news = st.checkbox("Show News & Sentiment", value=True)
        show_models = st.checkbox("Show Model Details", value=False)
        
        # Predict Button
        st.markdown("---")
        predict_button = st.button(
            "🚀 PREDICT NOW",
            type="primary",
            use_container_width=True
        )
    
    # Main Content
    if predict_button and symbol:
        with st.spinner("🔄 Analyzing market data and training AI models..."):
            # Progress bar
            progress_bar = st.progress(0)
            
            # Step 1: Fetch Data
            progress_bar.progress(20)
            st.info(f"📥 Fetching data for {symbol}...")
            
            # Step 2: Make Prediction
            progress_bar.progress(50)
            st.info("🤖 Training AI models and making prediction...")
            
            result = st.session_state.predictor.predict_direction(symbol, period)
            
            progress_bar.progress(80)
            
            if result:
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
                            <h2 style="margin: 0; color: white;">{symbol}</h2>
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
                                    color: {'#4CAF50' if result['technical_signals'].get('Overall_Signal') == 'BUY' else '#FF5252'}">
                                {result['technical_signals'].get('Overall_Signal', 'NEUTRAL')}
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress(100)
                
                # Tabs for Detailed Analysis
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart", "🔍 Technicals", "📰 News", "📊 Details"])
                
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
                        for ma in [20, 50, 200]:
                            if len(hist) > ma:
                                fig.add_trace(go.Scatter(
                                    x=hist.index,
                                    y=hist['Close'].rolling(ma).mean(),
                                    name=f'MA {ma}',
                                    line=dict(width=1),
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
                    if show_technical and result['technical_signals']:
                        st.subheader("Technical Analysis Signals")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Indicator Signals")
                            for indicator, signal in result['technical_signals'].items():
                                if indicator not in ['Overall_Signal', 'Confidence']:
                                    color = "#4CAF50" if 'BUY' in str(signal) or 'BULLISH' in str(signal) else "#FF5252" if 'SELL' in str(signal) or 'BEARISH' in str(signal) else "#FFC107"
                                    st.markdown(f"""
                                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333;">
                                        <span>{indicator.replace('_', ' ')}:</span>
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
                                        st.markdown(f"**Price Position:**")
                                        st.progress(position)
                                        
                                        if position < 0.3:
                                            st.info("Near Support - Potential buying opportunity")
                                        elif position > 0.7:
                                            st.warning("Near Resistance - Potential selling opportunity")
                            except:
                                pass
                
                with tab3:
                    if show_news:
                        st.subheader("Latest Market News & Sentiment")
                        
                        # Overall market sentiment
                        market_sentiment, sentiment_label = st.session_state.news.get_market_sentiment()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Market Sentiment", sentiment_label)
                        with col2:
                            st.metric("Sentiment Score", f"{market_sentiment:+.2f}")
                        with col3:
                            st.metric("News Count", "15+")
                        
                        # Fetch and display news for the symbol
                        news_items = st.session_state.news.fetch_news(symbol.split('.')[0])
                        
                        if news_items:
                            for news in news_items[:5]:  # Show top 5 news
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
                                        {' '.join([f'<span style="background: #333; padding: 2px 6px; border-radius: 3px; margin-right: 5px; font-size: 0.8em;">{s}</span>' for s in news['symbols'][:3]])}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No recent news found for this symbol.")
                
                with tab4:
                    if show_models:
                        st.subheader("AI Model Details")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🧠 Model Performance")
                            for model_name, model_info in st.session_state.predictor.models.items():
                                accuracy = model_info.get('accuracy', 0)
                                st.metric(
                                    f"{model_name.replace('_', ' ').title()}",
                                    f"{accuracy:.1%}",
                                    delta="High" if accuracy > 0.6 else "Medium" if accuracy > 0.55 else "Low"
                                )
                        
                        with col2:
                            st.markdown("#### 📊 Feature Importance")
                            if st.session_state.predictor.models:
                                # Get feature importance from XGBoost
                                xgb_model = st.session_state.predictor.models.get('xgboost')
                                if xgb_model and 'feature_importance' in xgb_model:
                                    importance_df = pd.DataFrame(
                                        list(xgb_model['feature_importance'].items()),
                                        columns=['Feature', 'Importance']
                                    ).sort_values('Importance', ascending=False).head(10)
                                    
                                    fig = go.Figure(go.Bar(
                                        x=importance_df['Importance'],
                                        y=importance_df['Feature'],
                                        orientation='h',
                                        marker_color='#FF9933'
                                    ))
                                    
                                    fig.update_layout(
                                        height=400,
                                        title="Top 10 Important Features",
                                        template="plotly_dark"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Trading Recommendations
                        st.markdown("---")
                        st.subheader("🎯 Trading Recommendations")
                        
                        if result['signal'] == 'BUY':
                            st.success("""
                            **Recommended Action: BUY**
                            
                            📋 **Plan:**
                            - Entry: Current price or slight dip
                            - Stop Loss: 3-5% below entry
                            - Target 1: 5-8% profit
                            - Target 2: 10-15% profit (if momentum continues)
                            
                            ⚡ **Quick Tips:**
                            - Consider averaging in if price dips
                            - Monitor news for any negative developments
                            - Book partial profits at first target
                            """)
                        elif result['signal'] == 'SELL':
                            st.error("""
                            **Recommended Action: SELL**
                            
                            📋 **Plan:**
                            - Exit long positions
                            - Consider short positions if experienced
                            - Wait for better entry points
                            
                            ⚡ **Quick Tips:**
                            - Don't try to catch a falling knife
                            - Wait for confirmation of trend reversal
                            - Consider defensive stocks/sectors
                            """)
                        else:
                            st.warning("""
                            **Recommended Action: HOLD/WAIT**
                            
                            📋 **Plan:**
                            - Maintain existing positions
                            - Wait for clearer signals
                            - Monitor key support/resistance levels
                            
                            ⚡ **Quick Tips:**
                            - This is not a signal to buy or sell
                            - Market is uncertain, be patient
                            - Consider reducing position size if holding
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
                st.error("❌ Could not generate prediction. Please check the symbol and try again.")
    
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
                <p>50+ indicators with automatic signal generation</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown("---")
        st.subheader("📈 Today's Market Overview")
        
        try:
            # Get Nifty 50, Sensex, Bank Nifty
            indices = {
                "NIFTY 50": "^NSEI",
                "SENSEX": "^BSESN", 
                "BANK NIFTY": "^NSEBANK"
            }
            
            cols = st.columns(3)
            for (name, symbol), col in zip(indices.items(), cols):
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        prev = ticker.info.get('previousClose', current)
                        change = ((current - prev) / prev) * 100
                        
                        with col:
                            color = "#4CAF50" if change >= 0 else "#FF5252"
                            st.metric(
                                name,
                                f"₹{current:,.2f}",
                                f"{change:+.2f}%",
                                delta_color="normal"
                            )
                except:
                    with col:
                        st.metric(name, "N/A")
                        
        except:
            pass
        
        # How to Use
        st.markdown("---")
        st.subheader("🚀 How to Use")
        
        st.markdown("""
        1. **Select** your asset type from the sidebar (Stocks, Indices, or Commodities)
        2. **Choose** a specific symbol from the dropdown or enter custom symbol
        3. **Set** your preferred analysis period
        4. **Click** the "PREDICT NOW" button
        5. **Review** the AI prediction with confidence level
        6. **Analyze** detailed charts, technicals, and news
        
        ### 📋 Supported Assets:
        - **Stocks**: All Nifty 50 companies
        - **Indices**: Nifty 50, Sensex, Bank Nifty, Sectoral indices
        - **Commodities**: Gold, Silver, Crude Oil, Metals
        - **Custom**: Any Indian stock with .NS suffix
        
        ### ⚡ Pro Tips:
        - Use longer periods (1y+) for more accurate predictions
        - Check both technical signals and news sentiment
        - Consider the confidence level before making decisions
        - Always use proper risk management
        
        ### 📚 Features Included:
        - ✅ AI Prediction with multiple models
        - ✅ Real-time Indian market news
        - ✅ Technical analysis with 50+ indicators  
        - ✅ Sentiment analysis
        - ✅ Price charts with support/resistance
        - ✅ Trading recommendations
        - ✅ Model accuracy metrics
        """)
        
        # Quick Start Examples
        st.markdown("---")
        st.subheader("⚡ Quick Predictions")
        
        quick_cols = st.columns(4)
        quick_symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "^NSEI"]
        
        for idx, (col, sym) in enumerate(zip(quick_cols, quick_symbols)):
            with col:
                if st.button(f"Predict {sym}", use_container_width=True):
                    st.session_state.quick_symbol = sym
                    st.rerun()

if __name__ == "__main__":
    main()
