import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# News
import feedparser
import random

# ============================================================================
# PAGE CONFIG
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
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #3c1a1a 0%, #5a2d2d 100%);
        border-left: 6px solid #FF5252;
    }
    
    .hold-signal {
        background: linear-gradient(135deg, #2d2d3c 0%, #3d3d5a 100%);
        border-left: 6px solid #FFC107;
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
    }
    
    .stock-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }
    
    .stock-table th {
        background: #1a1a2e;
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid #0f3460;
    }
    
    .stock-table td {
        padding: 10px;
        border-bottom: 1px solid #333;
    }
    
    .stock-table tr:hover {
        background: #1e2330;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# NIFTY 100 DATABASE
# ============================================================================

NIFTY_100 = {
    # Top 30 by weight
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
    "GRASIM": "GRASIM.NS",
    
    # Additional Nifty 100 stocks
    "INDUSIND BANK": "INDUSINDBK.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "HDFC LIFE": "HDFCLIFE.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "DIVIS LAB": "DIVISLAB.NS",
    "BAJAJ AUTO": "BAJAJ-AUTO.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "ADANI PORTS": "ADANIPORTS.NS",
    "HINDALCO": "HINDALCO.NS",
    "HERO MOTOCORP": "HEROMOTOCO.NS",
    "EICHER MOTORS": "EICHERMOT.NS",
    "DR REDDY": "DRREDDY.NS",
    "COAL INDIA": "COALINDIA.NS",
    "BPCL": "BPCL.NS",
    "SHREE CEMENT": "SHREECEM.NS",
    "JIO FINANCIAL": "JIOFIN.NS",
    "SIEMENS": "SIEMENS.NS",
    "HAL": "HAL.NS",
    "ABB": "ABB.NS",
    "PIDILITE": "PIDILITIND.NS",
    "DABUR": "DABUR.NS",
    "GODREJ CONSUMER": "GODREJCP.NS",
    "HAVELLS": "HAVELLS.NS",
    "ICICI LOMBARD": "ICICIGI.NS",
    "M&M": "M&M.NS",
    "VEDANTA": "VEDL.NS",
    "APOLLO HOSPITALS": "APOLLOHOSP.NS",
    "DLF": "DLF.NS",
    "ADANI GREEN": "ADANIGREEN.NS",
    "ZOMATO": "ZOMATO.NS",
    "BAJAJ HOLDINGS": "BAJAJHLDNG.NS",
    "PIRAMAL ENTERPRISES": "PEL.NS",
    "LUPIN": "LUPIN.NS",
    "TORRENT PHARMA": "TORNTPHARM.NS",
    "BHARAT ELECTRONICS": "BEL.NS",
    "LIC": "LICI.NS",
    "TIINDIA": "TIINDIA.NS",
    "TRENT": "TRENT.NS",
    "POLYCAB": "POLYCAB.NS",
    "MOTHERSON": "MOTHERSON.NS",
    "UNITED SPIRITS": "MCDOWELL-N.NS",
    "MAX HEALTH": "MAXHEALTH.NS",
    "AU SMALL FINANCE": "AUBANK.NS",
    "GAIL": "GAIL.NS",
    "SCHAEFFLER": "SCHAEFFLER.NS",
    "CONCOR": "CONCOR.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "INDIAN HOTELS": "INDHOTEL.NS",
    "BOSCH": "BOSCHLTD.NS",
    "DALMIABHA": "DALBHARAT.NS",
    "SRF": "SRF.NS",
    "NYKAA": "NYKAA.NS",
    "JINDAL STEEL": "JINDALSTEL.NS",
    "AMBUJA CEMENT": "AMBUJACEM.NS",
    "PATANJALI": "PATANJALI.NS",
    "BANK OF BARODA": "BANKBARODA.NS",
    "ABBOT INDIA": "ABBOTINDIA.NS",
    "ACC": "ACC.NS",
    "ICICI PRU LIFE": "ICICIPRULI.NS",
    "CANARA BANK": "CANBK.NS",
    "TVS MOTOR": "TVSMOTOR.NS",
    "CUMMINS": "CUMMINSIND.NS",
    "MARICO": "MARICO.NS",
    "ASHOK LEYLAND": "ASHOKLEY.NS",
    "BERGER PAINTS": "BERGEPAINT.NS",
    "PFC": "PFC.NS",
    "HINDPETRO": "HINDPETRO.NS",
    "GODREJ PROPERTIES": "GODREJPROP.NS",
    "BIOCON": "BIOCON.NS",
    "IRCTC": "IRCTC.NS",
    "INDUSTOWER": "INDUSTOWER.NS",
    "MUTHOOT FINANCE": "MUTHOOTFIN.NS",
    "TORRENT POWER": "TORNTPOWER.NS",
    "ASTRAL": "ASTRAL.NS",
    "PHOENIX MILLS": "PHOENIXLTD.NS",
    "MANKIND PHARMA": "MANKIND.NS",
    "IPCALAB": "IPCALAB.NS",
    "SUZLON": "SUZLON.NS",
    "JUBL FOOD": "JUBLFOOD.NS",
    "AJANTA PHARMA": "AJANTPHARM.NS",
    "ENDURANCE TECH": "ENDURANCE.NS",
    "KEI INDUSTRIES": "KEI.NS",
    "AUBANK": "AUBANK.NS",
    "ALKEM LAB": "ALKEM.NS",
    "CIPLA": "CIPLA.NS",
    "TATA POWER": "TATAPOWER.NS",
    "LTI MINDREE": "LTIM.NS",
    "RELIANCE INFRA": "RELINFRA.NS",
    "IDFC FIRST BANK": "IDFCFIRSTB.NS",
    "JUBILANT PHARMOVA": "JUBLPHARMA.NS"
}

INDIAN_INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN", 
    "BANK NIFTY": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY MIDCAP 50": "^NSEMDCP50",
    "NIFTY SMALLCAP 50": "^NSESC50"
}

# ============================================================================
# IMPROVED TECHNICAL INDICATORS WITH REALISTIC SIGNALS
# ============================================================================

def calculate_improved_indicators(df):
    """Calculate technical indicators with better signal logic"""
    df = df.copy()
    
    if len(df) < 50:
        return df
    
    # Price-based indicators
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # RSI with better calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume indicators
    if 'Volume' in df.columns:
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        # On Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    else:
        df['Volume_Ratio'] = 1
        df['OBV'] = 0
    
    # ATR for volatility
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    df['ATR_Percent'] = df['ATR'] / df['Close'] * 100
    
    # Price patterns
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Trend strength
    df['Trend_Strength'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean() * 100
    
    # Market regime (0: trending, 1: ranging)
    df['Market_Regime'] = (df['Trend_Strength'] < 2).astype(int)
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def generate_realistic_signals(df):
    """Generate more realistic trading signals - FIXED"""
    if df.empty or len(df) < 50:
        return {'Overall': 'HOLD', 'Confidence': 0.5, 'Reason': 'Insufficient data'}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    signals = {}
    
    # 1. Trend Analysis
    price_vs_sma20 = (latest['Close'] - latest['SMA_20']) / latest['SMA_20'] * 100
    price_vs_sma50 = (latest['Close'] - latest['SMA_50']) / latest['SMA_50'] * 100
    
    if price_vs_sma20 > 2 and price_vs_sma50 > 2:
        signals['Trend'] = 'STRONG_UPTREND'
        trend_score = 2
    elif price_vs_sma20 > 0 and price_vs_sma50 > 0:
        signals['Trend'] = 'UPTREND'
        trend_score = 1
    elif price_vs_sma20 < -2 and price_vs_sma50 < -2:
        signals['Trend'] = 'STRONG_DOWNTREND'
        trend_score = -2
    elif price_vs_sma20 < 0 and price_vs_sma50 < 0:
        signals['Trend'] = 'DOWNTREND'
        trend_score = -1
    else:
        signals['Trend'] = 'RANGING'
        trend_score = 0
    
    # 2. RSI Analysis
    rsi = latest['RSI']
    if rsi < 30:
        signals['RSI'] = 'OVERSOLD'
        rsi_score = 1  # Oversold is bullish
    elif rsi > 70:
        signals['RSI'] = 'OVERBOUGHT'
        rsi_score = -1  # Overbought is bearish
    elif 40 < rsi < 60:
        signals['RSI'] = 'NEUTRAL'
        rsi_score = 0
    elif rsi > 50:
        signals['RSI'] = 'BULLISH'
        rsi_score = 0.5
    else:
        signals['RSI'] = 'BEARISH'
        rsi_score = -0.5
    
    # 3. MACD Analysis
    macd_hist = latest['MACD_Histogram']
    macd_trend = 'BULLISH' if macd_hist > 0 else 'BEARISH'
    signals['MACD'] = macd_trend
    macd_score = 1 if macd_hist > 0 and macd_hist > prev['MACD_Histogram'] else -1 if macd_hist < 0 and macd_hist < prev['MACD_Histogram'] else 0
    
    # 4. Bollinger Bands
    bb_position = latest['BB_Position']
    if bb_position < 0.2:
        signals['BB'] = 'NEAR_SUPPORT'
        bb_score = 1
    elif bb_position > 0.8:
        signals['BB'] = 'NEAR_RESISTANCE'
        bb_score = -1
    else:
        signals['BB'] = 'MIDDLE'
        bb_score = 0
    
    # 5. Volume Analysis
    volume_ratio = latest['Volume_Ratio']
    if volume_ratio > 1.5:
        signals['Volume'] = 'HIGH_VOLUME'
        volume_score = 1 if latest['Returns'] > 0 else -1 if latest['Returns'] < 0 else 0
    elif volume_ratio < 0.5:
        signals['Volume'] = 'LOW_VOLUME'
        volume_score = 0
    else:
        signals['Volume'] = 'NORMAL'
        volume_score = 0
    
    # 6. Volatility Analysis
    atr_percent = latest['ATR_Percent']
    if atr_percent > 3:
        signals['Volatility'] = 'HIGH'
        vol_score = -0.5  # High volatility is risky
    elif atr_percent < 1:
        signals['Volatility'] = 'LOW'
        vol_score = 0.5
    else:
        signals['Volatility'] = 'NORMAL'
        vol_score = 0
    
    # Calculate overall score
    scores = [trend_score, rsi_score, macd_score, bb_score, volume_score, vol_score]
    valid_scores = [s for s in scores if s != 0]
    
    if not valid_scores:
        overall = 'HOLD'
        confidence = 0.5
    else:
        avg_score = sum(valid_scores) / len(valid_scores)
        
        if avg_score > 0.3:
            overall = 'BUY'
            confidence = min(0.3 + avg_score * 0.7, 0.95)
        elif avg_score < -0.3:
            overall = 'SELL'
            confidence = min(0.3 + abs(avg_score) * 0.7, 0.95)
        else:
            overall = 'HOLD'
            confidence = 0.5
    
    signals['Overall'] = overall
    signals['Confidence'] = confidence
    signals['Score'] = avg_score if valid_scores else 0
    
    return signals

# ============================================================================
# IMPROVED AI PREDICTOR WITH BETTER TRAINING
# ============================================================================

class ImprovedStockPredictor:
    """Improved stock predictor with realistic signals"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training_date = None
    
    def prepare_features_for_prediction(self, df, lookback_days=60):
        """Prepare features for single prediction"""
        if len(df) < lookback_days:
            return None
        
        # Get recent data
        recent = df.iloc[-lookback_days:]
        
        # Feature 1: Price momentum (last 5 vs last 20 days)
        if len(recent) >= 20:
            price_5 = recent['Close'].iloc[-5:].mean()
            price_20 = recent['Close'].iloc[-20:].mean()
            momentum = (price_5 - price_20) / price_20 * 100
        else:
            momentum = 0
        
        # Feature 2: RSI value
        rsi = recent['RSI'].iloc[-1] if 'RSI' in recent.columns else 50
        
        # Feature 3: MACD histogram trend
        if 'MACD_Histogram' in recent.columns:
            macd_now = recent['MACD_Histogram'].iloc[-1]
            macd_prev = recent['MACD_Histogram'].iloc[-5] if len(recent) >= 5 else 0
            macd_trend = 1 if macd_now > 0 and macd_now > macd_prev else -1 if macd_now < 0 and macd_now < macd_prev else 0
        else:
            macd_trend = 0
        
        # Feature 4: Bollinger Band position
        bb_pos = recent['BB_Position'].iloc[-1] if 'BB_Position' in recent.columns else 0.5
        
        # Feature 5: Volume trend
        if 'Volume_Ratio' in recent.columns:
            volume_now = recent['Volume_Ratio'].iloc[-1]
            volume_trend = 1 if volume_now > 1.2 else -1 if volume_now < 0.8 else 0
        else:
            volume_trend = 0
        
        # Feature 6: Volatility
        volatility = recent['ATR_Percent'].iloc[-1] if 'ATR_Percent' in recent.columns else 2
        
        # Feature 7: Market regime
        regime = recent['Market_Regime'].iloc[-1] if 'Market_Regime' in recent.columns else 0
        
        return np.array([[momentum, rsi, macd_trend, bb_pos, volume_trend, volatility, regime]])
    
    def train_on_multiple_stocks(self):
        """Train model on multiple stocks for better generalization"""
        try:
            # Select random stocks for training
            training_stocks = list(NIFTY_100.values())[:20]  # Train on 20 stocks
            
            all_features = []
            all_targets = []
            
            for symbol in training_stocks:
                try:
                    # Fetch data
                    stock = yf.Ticker(symbol)
                    df = stock.history(period="2y")
                    
                    if len(df) < 100:
                        continue
                    
                    # Calculate indicators
                    df = calculate_improved_indicators(df)
                    
                    # Prepare training samples
                    for i in range(100, len(df) - 10):
                        # Create features
                        window = df.iloc[i-60:i]
                        
                        # Similar features as prediction
                        if len(window) >= 20:
                            price_5 = window['Close'].iloc[-5:].mean()
                            price_20 = window['Close'].iloc[-20:].mean()
                            momentum = (price_5 - price_20) / price_20 * 100
                        else:
                            momentum = 0
                        
                        rsi = window['RSI'].iloc[-1] if 'RSI' in window.columns else 50
                        
                        if 'MACD_Histogram' in window.columns:
                            macd_now = window['MACD_Histogram'].iloc[-1]
                            macd_prev = window['MACD_Histogram'].iloc[-5] if len(window) >= 5 else 0
                            macd_trend = 1 if macd_now > 0 and macd_now > macd_prev else -1 if macd_now < 0 and macd_now < macd_prev else 0
                        else:
                            macd_trend = 0
                        
                        bb_pos = window['BB_Position'].iloc[-1] if 'BB_Position' in window.columns else 0.5
                        
                        if 'Volume_Ratio' in window.columns:
                            volume_now = window['Volume_Ratio'].iloc[-1]
                            volume_trend = 1 if volume_now > 1.2 else -1 if volume_now < 0.8 else 0
                        else:
                            volume_trend = 0
                        
                        volatility = window['ATR_Percent'].iloc[-1] if 'ATR_Percent' in window.columns else 2
                        regime = window['Market_Regime'].iloc[-1] if 'Market_Regime' in window.columns else 0
                        
                        features = [momentum, rsi, macd_trend, bb_pos, volume_trend, volatility, regime]
                        
                        # Target: Will price go up in next 5 days?
                        future_price = df['Close'].iloc[i+5]
                        current_price = df['Close'].iloc[i]
                        target = 1 if future_price > current_price else 0
                        
                        all_features.append(features)
                        all_targets.append(target)
                        
                except:
                    continue
            
            if len(all_features) < 100:
                return False
            
            # Convert to arrays
            X = np.array(all_features)
            y = np.array(all_targets)
            
            # Train-test split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle imbalance
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, self.model.predict(X_train_scaled))
            test_acc = accuracy_score(y_test, self.model.predict(X_test_scaled))
            
            self.is_trained = True
            self.last_training_date = datetime.now()
            
            return True
            
        except Exception as e:
            return False
    
    def predict(self, df):
        """Make prediction"""
        if not self.is_trained:
            # Train model if not trained
            success = self.train_on_multiple_stocks()
            if not success:
                return None
        
        try:
            # Prepare features
            features = self.prepare_features_for_prediction(df)
            if features is None:
                return None
            
            # Scale and predict
            features_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(features_scaled)[0]
            
            # Get prediction
            prediction = 1 if proba[1] > 0.55 else 0  # Higher threshold for confidence
            confidence = max(proba)
            
            # Adjust confidence based on probability difference
            prob_diff = abs(proba[1] - proba[0])
            confidence = 0.5 + prob_diff * 0.5  # Scale to 0.5-1.0 range
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probability_up': proba[1],
                'probability_down': proba[0]
            }
            
        except:
            return None

# ============================================================================
# MARKET SCANNER
# ============================================================================

def scan_market(limit=20):
    """Scan Nifty 100 stocks for opportunities"""
    results = []
    
    with st.spinner(f"🔍 Scanning top {limit} Nifty stocks..."):
        # Take first 'limit' stocks
        stocks_to_scan = list(NIFTY_100.items())[:limit]
        
        for name, symbol in stocks_to_scan:
            try:
                # Fetch data
                stock = yf.Ticker(symbol)
                df = stock.history(period="3mo")
                
                if len(df) < 30:
                    continue
                
                # Calculate indicators
                df = calculate_improved_indicators(df)
                
                # Generate signals
                signals = generate_realistic_signals(df)
                
                # Get price info
                current_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                results.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Price': current_price,
                    'Change %': change_pct,
                    'Signal': signals.get('Overall', 'HOLD'),
                    'Confidence': signals.get('Confidence', 0.5),
                    'RSI': df['RSI'].iloc[-1] if 'RSI' in df.columns else 50,
                    'Trend': signals.get('Trend', 'NEUTRAL')
                })
                
            except:
                continue
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("<h1 class='main-header'>🇮🇳 Nifty 100 Market Predictor AI</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = ImprovedStockPredictor()
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        
        page = st.radio(
            "Go to",
            ["Single Stock Analysis", "Market Scanner", "Indices Analysis"],
            index=0
        )
        
        st.markdown("---")
        
        if page == "Single Stock Analysis":
            st.markdown("### 🔍 Stock Selection")
            
            # Search with autocomplete
            search_term = st.text_input("Search stock...", "")
            
            if search_term:
                # Filter stocks based on search
                filtered_stocks = {k: v for k, v in NIFTY_100.items() 
                                 if search_term.upper() in k.upper() or search_term.upper() in v}
                if filtered_stocks:
                    selected_stock = st.selectbox(
                        "Select Stock",
                        list(filtered_stocks.keys())
                    )
                    symbol = filtered_stocks[selected_stock]
                else:
                    symbol = search_term.upper()
                    if not symbol.endswith('.NS') and not symbol.startswith('^'):
                        symbol += '.NS'
            else:
                selected_stock = st.selectbox(
                    "Select from Nifty 100",
                    list(NIFTY_100.keys())[:50]  # Show first 50 for performance
                )
                symbol = NIFTY_100[selected_stock]
            
            period = st.selectbox(
                "Analysis Period",
                ["1mo", "3mo", "6mo", "1y"],
                index=1
            )
            
            if st.button("🚀 ANALYZE STOCK", type="primary", use_container_width=True):
                st.session_state.analyze_symbol = symbol
                st.session_state.analyze_period = period
                st.rerun()
        
        elif page == "Market Scanner":
            st.markdown("### ⚡ Quick Scanner")
            
            scanner_type = st.selectbox(
                "Scan for",
                ["All Signals", "Strong BUY Only", "Strong SELL Only", "Oversold", "Overbought"]
            )
            
            limit = st.slider("Number of stocks", 10, 50, 20)
            
            if st.button("🔍 SCAN MARKET", type="primary", use_container_width=True):
                st.session_state.scan_type = scanner_type
                st.session_state.scan_limit = limit
                st.rerun()
        
        else:  # Indices Analysis
            st.markdown("### 📈 Index Selection")
            
            selected_index = st.selectbox(
                "Select Index",
                list(INDIAN_INDICES.keys())
            )
            symbol = INDIAN_INDICES[selected_index]
            
            period = st.selectbox(
                "Analysis Period",
                ["1mo", "3mo", "6mo", "1y"],
                index=1
            )
            
            if st.button("📊 ANALYZE INDEX", type="primary", use_container_width=True):
                st.session_state.analyze_symbol = symbol
                st.session_state.analyze_period = period
                st.rerun()
    
    # Main content
    if page == "Single Stock Analysis" and 'analyze_symbol' in st.session_state:
        symbol = st.session_state.analyze_symbol
        period = st.session_state.analyze_period
        
        st.markdown(f"### 📊 Analyzing: {symbol}")
        
        # Fetch and analyze
        with st.spinner("📥 Fetching market data..."):
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(period=period)
                
                if df.empty:
                    st.error("❌ No data found!")
                    return
                
                # Calculate indicators
                df = calculate_improved_indicators(df)
                
                # Show basic info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = df['Close'].iloc[-1]
                    st.metric("Current Price", f"₹{current_price:.2f}")
                
                with col2:
                    if len(df) > 1:
                        change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                        st.metric("Today's Change", f"{change:+.2f}%")
                    else:
                        st.metric("Today's Change", "N/A")
                
                with col3:
                    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                    st.metric("RSI", f"{rsi:.1f}")
                
                with col4:
                    volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
                    st.metric("Volume", f"{volume:,.0f}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return
        
        # Get signals
        with st.spinner("🤖 Analyzing signals..."):
            # Technical signals
            tech_signals = generate_realistic_signals(df)
            
            # AI prediction
            ai_prediction = st.session_state.predictor.predict(df)
            
            # Combine results
            if ai_prediction:
                # Use AI if available
                direction = "UP" if ai_prediction['prediction'] == 1 else "DOWN"
                confidence = ai_prediction['confidence']
                signal = "BUY" if ai_prediction['prediction'] == 1 else "SELL"
                method = "AI + Technical Analysis"
                ai_confidence = ai_prediction['confidence']
            else:
                # Use technical only
                direction = "UP" if tech_signals.get('Overall') == 'BUY' else "DOWN"
                confidence = tech_signals.get('Confidence', 0.5)
                signal = tech_signals.get('Overall', 'HOLD')
                method = "Technical Analysis Only"
                ai_confidence = None
            
            # Price change
            if len(df) > 1:
                price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            else:
                price_change = 0
            
            # Store result
            result = {
                'symbol': symbol,
                'prediction': direction,
                'signal': signal,
                'confidence': confidence,
                'price': df['Close'].iloc[-1],
                'change_pct': price_change,
                'technical_signals': tech_signals,
                'method': method,
                'ai_confidence': ai_confidence,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.last_result = result
        
        # Display results
        if st.session_state.last_result:
            result = st.session_state.last_result
            
            st.markdown("---")
            
            # Prediction card
            signal_color = "#4CAF50" if result['signal'] == 'BUY' else "#FF5252" if result['signal'] == 'SELL' else "#FFC107"
            
            st.markdown(f"""
            <div class="prediction-card" style="border-left: 6px solid {signal_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; color: white;">{result['symbol']}</h2>
                        <p style="color: #aaa; margin: 5px 0;">{result['method']}</p>
                    </div>
                    <div style="text-align: right;">
                        <h1 style="margin: 0; font-size: 3rem; color: {signal_color};">{result['signal']}</h1>
                        <p style="color: #aaa;">Expected: {result['prediction']}</p>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                        <h3 style="margin: 0 0 10px 0;">Confidence: {result['confidence']:.1%}</h3>
                        <div style="height: 10px; background: rgba(255,255,255,0.2); border-radius: 5px; overflow: hidden;">
                            <div style="height: 100%; width: {result['confidence']*100}%; 
                                     background: {signal_color}; 
                                     border-radius: 5px;"></div>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 20px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div class="metric-card">
                        <h4 style="margin: 0; color: #aaa;">Current Price</h4>
                        <p style="font-size: 1.5rem; margin: 10px 0; color: white;">₹{result['price']:.2f}</p>
                    </div>
                    <div class="metric-card">
                        <h4 style="margin: 0; color: #aaa;">Today's Change</h4>
                        <p style="font-size: 1.5rem; margin: 10px 0; 
                                color: {'#4CAF50' if result['change_pct'] > 0 else '#FF5252'}">
                            {result['change_pct']:+.2f}%
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed analysis
            with st.expander("📊 Detailed Technical Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Indicator Status")
                    for indicator, value in result['technical_signals'].items():
                        if indicator not in ['Overall', 'Confidence', 'Score']:
                            if 'UPTREND' in str(value) or 'BULLISH' in str(value) or 'SUPPORT' in str(value):
                                color = "#4CAF50"
                            elif 'DOWNTREND' in str(value) or 'BEARISH' in str(value) or 'RESISTANCE' in str(value):
                                color = "#FF5252"
                            else:
                                color = "#FFC107"
                            
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333;">
                                <span>{indicator}:</span>
                                <span style="color: {color}; font-weight: bold;">{value}</span>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 🎯 Key Levels")
                    
                    # Calculate support and resistance
                    recent = df.tail(20)
                    if not recent.empty:
                        current = recent['Close'].iloc[-1]
                        support = recent['Low'].min()
                        resistance = recent['High'].max()
                        
                        st.metric("Current", f"₹{current:.2f}")
                        st.metric("Support", f"₹{support:.2f}")
                        st.metric("Resistance", f"₹{resistance:.2f}")
                        
                        # Position in range
                        if resistance > support:
                            position = (current - support) / (resistance - support)
                            st.markdown(f"**Position in Range:** {position:.0%}")
                            st.progress(position)
            
            # Chart
            st.markdown("#### 📊 Price Chart")
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))
            
            # Add indicators
            if 'SMA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ))
            
            if 'BB_Upper' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash')
                ))
            
            fig.update_layout(
                title=f"{symbol} Price Action",
                yaxis_title="Price (₹)",
                template="plotly_dark",
                height=500,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Market Scanner" and 'scan_type' in st.session_state:
        st.markdown("### 🔍 Nifty 100 Market Scanner")
        
        # Run scanner
        scanner_results = scan_market(st.session_state.scan_limit)
        
        if not scanner_results.empty:
            # Filter based on scan type
            if st.session_state.scan_type == "Strong BUY Only":
                filtered = scanner_results[scanner_results['Signal'] == 'BUY']
                filtered = filtered[filtered['Confidence'] > 0.7]
            elif st.session_state.scan_type == "Strong SELL Only":
                filtered = scanner_results[scanner_results['Signal'] == 'SELL']
                filtered = filtered[filtered['Confidence'] > 0.7]
            elif st.session_state.scan_type == "Oversold":
                filtered = scanner_results[scanner_results['RSI'] < 30]
            elif st.session_state.scan_type == "Overbought":
                filtered = scanner_results[scanner_results['RSI'] > 70]
            else:
                filtered = scanner_results
            
            st.markdown(f"**Found {len(filtered)} opportunities**")
            
            # Display results
            for _, row in filtered.iterrows():
                signal_color = "#4CAF50" if row['Signal'] == 'BUY' else "#FF5252" if row['Signal'] == 'SELL' else "#FFC107"
                change_color = "#4CAF50" if row['Change %'] > 0 else "#FF5252"
                
                st.markdown(f"""
                <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {signal_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: white;">{row['Name']}</h4>
                            <p style="margin: 5px 0; color: #aaa;">{row['Symbol']}</p>
                        </div>
                        <div style="text-align: right;">
                            <h3 style="margin: 0; color: {signal_color};">{row['Signal']}</h3>
                            <p style="margin: 5px 0; color: #aaa;">Confidence: {row['Confidence']:.0%}</p>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                        <span style="color: white;">₹{row['Price']:.2f}</span>
                        <span style="color: {change_color};">{row['Change %']:+.2f}%</span>
                        <span style="color: #FFC107;">RSI: {row['RSI']:.1f}</span>
                        <span style="color: #aaa;">{row['Trend']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No opportunities found with current filters.")
    
    elif page == "Indices Analysis" and 'analyze_symbol' in st.session_state:
        # Similar to single stock analysis but for indices
        symbol = st.session_state.analyze_symbol
        period = st.session_state.analyze_period
        
        st.markdown(f"### 📈 Analyzing Index: {symbol}")
        
        # Similar analysis flow as single stock
        # (Implementation similar to single stock analysis)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 40px 20px;'>
            <h2>📊 Nifty 100 Market Intelligence Platform</h2>
            <p style='color: #aaa; font-size: 1.1rem; max-width: 800px; margin: 20px auto;'>
                Advanced AI-powered analysis for all Nifty 100 stocks and major Indian indices.
                Get realistic BUY/SELL signals with confidence scoring.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Scan Top 20 Stocks", use_container_width=True):
                st.session_state.page = "Market Scanner"
                st.session_state.scan_type = "All Signals"
                st.session_state.scan_limit = 20
                st.rerun()
        
        with col2:
            if st.button("📈 Analyze RELIANCE", use_container_width=True):
                st.session_state.page = "Single Stock Analysis"
                st.session_state.analyze_symbol = "RELIANCE.NS"
                st.session_state.analyze_period = "3mo"
                st.rerun()
        
        with col3:
            if st.button("📊 Analyze NIFTY 50", use_container_width=True):
                st.session_state.page = "Indices Analysis"
                st.session_state.analyze_symbol = "^NSEI"
                st.session_state.analyze_period = "3mo"
                st.rerun()
        
        # Stats
        st.markdown("---")
        st.markdown("### 📋 Platform Features")
        
        features = [
            ("🤖 AI Prediction", "Trained on multiple stocks for realistic signals"),
            ("📊 Nifty 100 Coverage", "All top Indian stocks"),
            ("🔍 Market Scanner", "Quick scanning for opportunities"),
            ("📈 Technical Analysis", "Advanced indicators with realistic logic"),
            ("🎯 Confidence Scoring", "Not just BUY/SELL - shows confidence levels"),
            ("⚡ Real-time Data", "Live market data from Yahoo Finance")
        ]
        
        cols = st.columns(3)
        for idx, (title, desc) in enumerate(features):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style='padding: 15px; background: #1a1a2e; border-radius: 10px; margin-bottom: 10px;'>
                    <h4 style='margin: 0 0 10px 0;'>{title}</h4>
                    <p style='color: #aaa; margin: 0; font-size: 0.9rem;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
