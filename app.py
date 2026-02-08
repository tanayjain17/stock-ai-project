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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INDIAN MARKET DATABASE
# ============================================================================

INDIAN_STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS", 
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "INFOSYS": "INFY.NS",
    "ITC": "ITC.NS",
    "SBIN": "SBIN.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "KOTAK BANK": "KOTAKBANK.NS",
    "AXIS BANK": "AXISBANK.NS",
    "MARUTI": "MARUTI.NS",
    "TITAN": "TITAN.NS"
}

INDIAN_INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN", 
    "BANK NIFTY": "^NSEBANK"
}

# ============================================================================
# SIMPLE TECHNICAL INDICATORS
# ============================================================================

def calculate_simple_indicators(df):
    """Calculate basic technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
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
    
    # Volume
    if 'Volume' in df.columns:
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    else:
        df['Volume_Ratio'] = 1
    
    # Price change
    df['Returns'] = df['Close'].pct_change()
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def generate_technical_signals(df):
    """Generate simple trading signals"""
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    signals = {
        'RSI': 'BUY' if latest['RSI'] < 30 else 'SELL' if latest['RSI'] > 70 else 'NEUTRAL',
        'MACD': 'BUY' if latest['MACD'] > latest['MACD_Signal'] else 'SELL',
        'Trend': 'BUY' if latest['Close'] > latest['SMA_20'] else 'SELL',
        'BB': 'BUY' if latest['Close'] < latest['BB_Lower'] else 'SELL' if latest['Close'] > latest['BB_Upper'] else 'NEUTRAL'
    }
    
    # Count signals
    buy_count = sum([1 for s in signals.values() if s == 'BUY'])
    sell_count = sum([1 for s in signals.values() if s == 'SELL'])
    
    if buy_count > sell_count:
        overall = 'BUY'
        confidence = buy_count / 4
    elif sell_count > buy_count:
        overall = 'SELL'
        confidence = sell_count / 4
    else:
        overall = 'HOLD'
        confidence = 0.5
    
    signals['Overall'] = overall
    signals['Confidence'] = confidence
    
    return signals

# ============================================================================
# SIMPLE AI PREDICTOR
# ============================================================================

class SimpleStockPredictor:
    """Simple but reliable stock predictor"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        if len(df) < 100:
            return None, None
        
        # Create features
        features = []
        targets = []
        
        for i in range(60, len(df) - 5):
            # Technical features
            window = df.iloc[i-60:i]
            
            # Basic features
            rsi = window['RSI'].iloc[-1] if 'RSI' in window.columns else 50
            macd_diff = (window['MACD'].iloc[-1] - window['MACD_Signal'].iloc[-1]) if 'MACD' in window.columns else 0
            price_sma_ratio = window['Close'].iloc[-1] / window['SMA_20'].iloc[-1] if 'SMA_20' in window.columns else 1
            volume_ratio = window['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in window.columns else 1
            
            # Price momentum
            returns_5 = window['Close'].iloc[-1] / window['Close'].iloc[-6] - 1 if len(window) >= 6 else 0
            
            features.append([rsi, macd_diff, price_sma_ratio, volume_ratio, returns_5])
            
            # Target: Will price go up in next 5 days?
            future_price = df['Close'].iloc[i+5]
            current_price = df['Close'].iloc[i]
            target = 1 if future_price > current_price else 0
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def train_model(self, df):
        """Train the model"""
        try:
            X, y = self.prepare_training_data(df)
            
            if X is None or len(X) < 50:
                return False
            
            # Split data
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, self.model.predict(X_train_scaled))
            test_acc = accuracy_score(y_test, self.model.predict(X_test_scaled))
            
            self.is_trained = True
            return True
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return False
    
    def predict(self, df):
        """Make prediction"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            # Get latest features
            latest = df.iloc[-60:]
            
            rsi = latest['RSI'].iloc[-1] if 'RSI' in latest.columns else 50
            macd_diff = (latest['MACD'].iloc[-1] - latest['MACD_Signal'].iloc[-1]) if 'MACD' in latest.columns else 0
            price_sma_ratio = latest['Close'].iloc[-1] / latest['SMA_20'].iloc[-1] if 'SMA_20' in latest.columns else 1
            volume_ratio = latest['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in latest.columns else 1
            returns_5 = latest['Close'].iloc[-1] / latest['Close'].iloc[-6] - 1 if len(latest) >= 6 else 0
            
            features = np.array([[rsi, macd_diff, price_sma_ratio, volume_ratio, returns_5]])
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probability
            proba = self.model.predict_proba(features_scaled)[0]
            prediction = 1 if proba[1] > 0.5 else 0
            confidence = max(proba)
            
            return {
                'prediction': prediction,  # 1 for UP, 0 for DOWN
                'confidence': confidence,
                'probability_up': proba[1],
                'probability_down': proba[0]
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

# ============================================================================
# NEWS FETCHER
# ============================================================================

def fetch_news(symbol):
    """Fetch news for a symbol"""
    try:
        # Clean symbol for news search
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '').replace('^', '')
        
        # Try multiple news sources
        sources = [
            f"https://www.moneycontrol.com/rss/{clean_symbol.lower()}news.xml",
            "https://www.moneycontrol.com/rss/marketreports.xml",
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
        ]
        
        all_news = []
        
        for source in sources:
            try:
                feed = feedparser.parse(source)
                for entry in feed.entries[:5]:
                    news_item = {
                        'title': entry.title,
                        'link': entry.link,
                        'source': 'MoneyControl' if 'moneycontrol' in source else 'Economic Times',
                        'published': entry.published[:16] if entry.published else '',
                        'summary': entry.get('summary', '')[:200]
                    }
                    all_news.append(news_item)
            except:
                continue
        
        return all_news[:10]  # Return top 10
        
    except Exception as e:
        return []

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("<h1 class='main-header'>🇮🇳 Indian Market Predictor AI</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = SimpleStockPredictor()
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Select Asset")
        
        asset_type = st.radio(
            "Asset Type",
            ["Stocks", "Indices"],
            index=0
        )
        
        if asset_type == "Stocks":
            selected = st.selectbox(
                "Select Stock",
                list(INDIAN_STOCKS.keys())
            )
            symbol = INDIAN_STOCKS[selected]
        else:
            selected = st.selectbox(
                "Select Index", 
                list(INDIAN_INDICES.keys())
            )
            symbol = INDIAN_INDICES[selected]
        
        # Custom symbol
        custom_symbol = st.text_input("Or enter custom symbol:", "")
        if custom_symbol:
            symbol = custom_symbol.upper()
            if not symbol.endswith('.NS') and not symbol.startswith('^'):
                symbol += '.NS'
        
        period = st.selectbox(
            "Data Period",
            ["3mo", "6mo", "1y", "2y"],
            index=1
        )
        
        st.markdown("---")
        
        if st.button("🚀 PREDICT NOW", type="primary", use_container_width=True):
            st.session_state.predict_symbol = symbol
            st.session_state.predict_period = period
            st.rerun()
    
    # Main content
    if 'predict_symbol' in st.session_state:
        symbol = st.session_state.predict_symbol
        period = st.session_state.predict_period
        
        st.markdown(f"### 📊 Analyzing: {symbol}")
        
        # Fetch data
        with st.spinner("📥 Fetching market data..."):
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(period=period)
                
                if df.empty:
                    st.error("❌ No data found for this symbol!")
                    return
                
                # Calculate indicators
                df = calculate_simple_indicators(df)
                
                # Show basic info
                col1, col2, col3 = st.columns(3)
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
                    st.metric("Data Points", len(df))
                
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                return
        
        # Train model and predict
        with st.spinner("🤖 Training AI model..."):
            if not st.session_state.predictor.is_trained:
                success = st.session_state.predictor.train_model(df)
                if not success:
                    st.warning("⚠️ Using technical analysis only (insufficient data for AI training)")
            
            # Get technical signals
            tech_signals = generate_technical_signals(df)
            
            # Try AI prediction
            ai_prediction = None
            if st.session_state.predictor.is_trained:
                ai_prediction = st.session_state.predictor.predict(df)
            
            # Combine results
            if ai_prediction:
                # Use AI prediction
                direction = "UP" if ai_prediction['prediction'] == 1 else "DOWN"
                confidence = ai_prediction['confidence']
                signal = "BUY" if ai_prediction['prediction'] == 1 else "SELL"
                method = "AI + Technical Analysis"
            else:
                # Use technical analysis only
                direction = "UP" if tech_signals.get('Overall') == 'BUY' else "DOWN"
                confidence = tech_signals.get('Confidence', 0.5)
                signal = tech_signals.get('Overall', 'HOLD')
                method = "Technical Analysis Only"
            
            # Prepare result
            result = {
                'symbol': symbol,
                'prediction': direction,
                'signal': signal,
                'confidence': confidence,
                'price': df['Close'].iloc[-1],
                'change_pct': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0,
                'technical_signals': tech_signals,
                'method': method,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.last_result = result
        
        # Display results
        if st.session_state.last_result:
            result = st.session_state.last_result
            
            st.markdown("---")
            
            # Prediction card
            if result['signal'] == 'BUY':
                card_class = "buy-signal"
                emoji = "🚀"
                color = "#4CAF50"
            elif result['signal'] == 'SELL':
                card_class = "sell-signal"
                emoji = "⚠️"
                color = "#FF5252"
            else:
                card_class = "hold-signal"
                emoji = "⏸️"
                color = "#FFC107"
            
            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; color: white;">{result['symbol']}</h2>
                        <p style="color: #aaa; margin: 5px 0;">Prediction Method: {result['method']}</p>
                    </div>
                    <div style="text-align: right;">
                        <h1 style="margin: 0; font-size: 3rem; color: {color};">{emoji} {result['signal']}</h1>
                        <p style="color: #aaa;">Direction: {result['prediction']}</p>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                        <h3 style="margin: 0 0 10px 0;">Confidence: {result['confidence']:.1%}</h3>
                        <div style="height: 10px; background: rgba(255,255,255,0.2); border-radius: 5px; overflow: hidden;">
                            <div style="height: 100%; width: {result['confidence']*100}%; 
                                     background: {color}; 
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
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["📈 Chart", "🔍 Analysis", "📰 News"])
            
            with tab1:
                # Price chart
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ))
                
                # Add moving averages
                if 'SMA_20' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['SMA_20'],
                        name='SMA 20',
                        line=dict(color='orange', width=1)
                    ))
                
                if 'SMA_50' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['SMA_50'],
                        name='SMA 50',
                        line=dict(color='blue', width=1)
                    ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    yaxis_title="Price (₹)",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=500,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Technical analysis
                st.subheader("Technical Analysis Signals")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Indicator Status")
                    for indicator, signal in result['technical_signals'].items():
                        if indicator not in ['Overall', 'Confidence']:
                            sig_color = "#4CAF50" if signal == 'BUY' else "#FF5252" if signal == 'SELL' else "#FFC107"
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333;">
                                <span>{indicator}:</span>
                                <span style="color: {sig_color}; font-weight: bold;">{signal}</span>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 📈 Key Metrics")
                    
                    # Calculate some key levels
                    recent_data = df.tail(20)
                    
                    if not recent_data.empty:
                        current_price = recent_data['Close'].iloc[-1]
                        support = recent_data['Low'].min()
                        resistance = recent_data['High'].max()
                        
                        st.metric("Current Price", f"₹{current_price:.2f}")
                        st.metric("Support Level", f"₹{support:.2f}")
                        st.metric("Resistance Level", f"₹{resistance:.2f}")
                        
                        # RSI
                        if 'RSI' in recent_data.columns:
                            rsi_value = recent_data['RSI'].iloc[-1]
                            st.metric("RSI", f"{rsi_value:.1f}")
                            
                            if rsi_value < 30:
                                st.info("RSI indicates OVERSOLD condition")
                            elif rsi_value > 70:
                                st.warning("RSI indicates OVERBOUGHT condition")
            
            with tab3:
                # News
                st.subheader("📰 Latest Market News")
                
                with st.spinner("Fetching news..."):
                    news_items = fetch_news(symbol)
                
                if news_items:
                    for news in news_items[:5]:
                        st.markdown(f"""
                        <div class="news-card">
                            <div style="margin-bottom: 8px;">
                                <small style="color: #888;">{news['source']} • {news['published']}</small>
                            </div>
                            <a href="{news['link']}" target="_blank" style="color: white; text-decoration: none; font-weight: 500;">
                                {news['title']}
                            </a>
                            <div style="margin-top: 8px;">
                                <small style="color: #aaa;">{news['summary']}...</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent news found for this symbol.")
            
            # Trading recommendations
            st.markdown("---")
            st.subheader("🎯 Trading Recommendations")
            
            if result['signal'] == 'BUY':
                st.success(f"""
                **Recommended Action: BUY**
                
                **Confidence Level: {result['confidence']:.0%}**
                
                📋 **Suggested Plan:**
                - Entry Price: ₹{result['price']:.2f}
                - Stop Loss: ₹{result['price'] * 0.97:.2f} (3% below)
                - Target 1: ₹{result['price'] * 1.05:.2f} (5% profit)
                - Target 2: ₹{result['price'] * 1.10:.2f} (10% profit)
                
                ⚠️ **Risk Management:**
                - Never risk more than 2% of your capital on a single trade
                - Use proper position sizing
                - Monitor the trade daily
                """)
            elif result['signal'] == 'SELL':
                st.error(f"""
                **Recommended Action: SELL**
                
                **Confidence Level: {result['confidence']:.0%}**
                
                📋 **Suggested Plan:**
                - Consider exiting long positions
                - Wait for better entry points
                - Look for confirmation signals
                
                ⚠️ **Caution:**
                - Market shows bearish signs
                - Consider defensive strategies
                - Wait for trend confirmation before shorting
                """)
            
            # Disclaimer
            st.markdown("---")
            st.warning("""
            **⚠️ IMPORTANT DISCLAIMER:**
            
            This is for educational purposes only. Not financial advice.
            
            - Past performance doesn't guarantee future results
            - Always do your own research
            - Consult a financial advisor before investing
            - Invest only what you can afford to lose
            """)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 40px 20px;'>
            <h2>🎯 AI-Powered Indian Market Predictions</h2>
            <p style='color: #aaa; font-size: 1.1rem; max-width: 800px; margin: 20px auto;'>
                Get accurate BUY/SELL signals for Indian stocks and indices using 
                AI models and technical analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick start
        st.markdown("### ⚡ Quick Start")
        
        col1, col2, col3, col4 = st.columns(4)
        
        quick_symbols = [
            ("RELIANCE.NS", "RELIANCE"),
            ("TCS.NS", "TCS"),
            ("HDFCBANK.NS", "HDFC BANK"), 
            ("^NSEI", "NIFTY 50")
        ]
        
        for idx, (symbol, name) in enumerate(quick_symbols):
            with [col1, col2, col3, col4][idx]:
                if st.button(f"Predict {name}", use_container_width=True):
                    st.session_state.predict_symbol = symbol
                    st.session_state.predict_period = "6mo"
                    st.rerun()
        
        # Features
        st.markdown("---")
        st.subheader("✨ Features")
        
        features = [
            ("🤖 AI Prediction", "Uses machine learning to predict market direction"),
            ("📊 Technical Analysis", "20+ indicators with automatic signal generation"),
            ("📰 Market News", "Latest news from Indian financial sources"),
            ("🎯 Clear Signals", "Simple BUY/SELL/HOLD recommendations"),
            ("📈 Real-time Charts", "Interactive charts with technical indicators"),
            ("⚡ Fast & Reliable", "Quick predictions with confidence scores")
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
