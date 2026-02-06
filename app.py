import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from textblob import TextBlob
import feedparser
import urllib.parse
from datetime import datetime
import time
import tensorflow as tf

# --- 0. SEED SETTING (Fixes Random Flipping) ---
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Market Pulse AI", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .metric-card {
        background-color: #1e2330;
        border: 1px solid #2a2f3d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .news-card {
        background-color: #151922;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid #2962ff;
        transition: transform 0.2s;
    }
    .news-card:hover { transform: translateX(5px); background-color: #1a1f2b; }
    section[data-testid="stSidebar"] { background-color: #12151e; border-right: 1px solid #2a2f3d; }
</style>
""", unsafe_allow_html=True)

# --- 2. ASSET LISTS ---
NIFTY_100_TICKERS = {
    "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS", "L&T": "LT.NS",
    "Kotak Bank": "KOTAKBANK.NS", "Axis Bank": "AXISBANK.NS", "Tata Motors": "TATAMOTORS.NS",
    "Maruti Suzuki": "MARUTI.NS", "Bajaj Finance": "BAJFINANCE.NS", "Sun Pharma": "SUNPHARMA.NS",
    "Titan": "TITAN.NS", "NTPC": "NTPC.NS", "Power Grid": "POWERGRID.NS", "ONGC": "ONGC.NS",
    "Zomato": "ZOMATO.NS", "Adani Ent": "ADANIENT.NS", "DLF": "DLF.NS", "HAL": "HAL.NS"
}

ETFS_MFS = {
    "Nifty 50 ETF": "NIFTYBEES.NS", "Bank Nifty ETF": "BANKBEES.NS", "IT Tech ETF": "ITBEES.NS",
    "Gold ETF": "GOLDBEES.NS", "Silver ETF": "SILVERBEES.NS", "US Nasdaq 100": "MON100.NS"
}

COMMODITIES_GLOBAL = {
    "Gold (Global)": "GC=F", "Silver (Global)": "SI=F", "Crude Oil (WTI)": "CL=F"
}

# --- 3. HELPER FUNCTIONS ---

def get_live_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.fast_info.last_price
        prev_close = stock.fast_info.previous_close
        return price, price - prev_close, (price - prev_close) / prev_close * 100
    except:
        return 0.0, 0.0, 0.0

def get_news_with_sources(query_term):
    try:
        safe_query = urllib.parse.quote(query_term)
        rss_url = f"https://news.google.com/rss/search?q={safe_query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        news_items = []
        for entry in feed.entries:
            blob = TextBlob(entry.title)
            pol = blob.sentiment.polarity
            sentiment = "🟢 Bullish" if pol > 0.05 else "🔴 Bearish" if pol < -0.05 else "⚪ Neutral"
            timestamp = time.mktime(entry.published_parsed) if 'published_parsed' in entry else 0
            news_items.append({
                "title": entry.title, "link": entry.link, "source": entry.get('source', {}).get('title', 'Google News'),
                "date_str": entry.get("published", "")[:16], "timestamp": timestamp, "sentiment": sentiment
            })
        return sorted(news_items, key=lambda x: x['timestamp'], reverse=True)[:10]
    except:
        return []

def add_technical_indicators(df):
    # SMA (Simple Moving Average)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    # EMA (Exponential Moving Average - Reacts faster)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    return df.dropna() # Drop NaNs created by indicators

# --- 4. SIDEBAR ---
st.sidebar.title("🔍 Market Scanner")
view_mode = st.sidebar.radio("Navigation", ["🏠 Market Dashboard", "📈 Stock Analyzer", "🏦 ETFs & Funds", "🛢️ Global Commodities"])

selected_ticker = "RELIANCE.NS"
if view_mode == "📈 Stock Analyzer":
    stock_name = st.sidebar.selectbox("Top Stocks", list(NIFTY_100_TICKERS.keys()))
    selected_ticker = NIFTY_100_TICKERS[stock_name]
    if st.sidebar.radio("Exchange", ["NSE", "BSE"], horizontal=True) == "BSE":
        selected_ticker = selected_ticker.replace(".NS", ".BO")
    custom = st.sidebar.text_input("Or Type Ticker (e.g. IRFC)")
    if custom: selected_ticker = f"{custom.upper()}.NS"
elif view_mode == "🏦 ETFs & Funds":
    selected_ticker = ETFS_MFS[st.sidebar.selectbox("Select ETF", list(ETFS_MFS.keys()))]
elif view_mode == "🛢️ Global Commodities":
    selected_ticker = COMMODITIES_GLOBAL[st.sidebar.selectbox("Select Commodity", list(COMMODITIES_GLOBAL.keys()))]

if view_mode != "🏠 Market Dashboard":
    st.sidebar.markdown("---")
    chart_range = st.sidebar.selectbox("Chart History", ["1 Day", "1 Week", "1 Month", "6 Months", "1 Year", "5 Years"], index=2)

# --- 5. DASHBOARD PAGE ---
if view_mode == "🏠 Market Dashboard":
    st.title("🌏 Indian Markets Live")
    col1, col2, col3, col4 = st.columns(4)
    for n, s, c in [("NIFTY 50", "^NSEI", col1), ("SENSEX", "^BSESN", col2), ("BANK NIFTY", "^NSEBANK", col3), ("INDIA VIX", "^INDIAVIX", col4)]:
        p, ch, pc = get_live_price(s)
        clr = "#00e676" if ch >= 0 else "#ff1744"
        with c:
            st.markdown(f'<div class="metric-card" style="border-top: 3px solid {clr};"><div>{n}</div><div style="font-size:24px; font-weight:bold;">{p:,.2f}</div><div style="color:{clr};">{ch:+.2f} ({pc:+.2f}%)</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📰 Market News")
    for n in get_news_with_sources("Indian Stock Market"):
        st.markdown(f'<div class="news-card"><div style="display:flex; justify-content:space-between; color:#aaa; font-size:12px;"><span>{n["source"]}</span><span>{n["date_str"]}</span></div><a href="{n["link"]}" target="_blank" style="color:white; font-weight:bold;">{n["title"]}</a><div style="font-size:12px; margin-top:5px;">{n["sentiment"]}</div></div>', unsafe_allow_html=True)

# --- 6. ANALYSIS PAGE ---
else:
    st.title(f"⚡ {selected_ticker} Pro Analysis")
    lp, lc, lpct = get_live_price(selected_ticker)
    clr = "#00e676" if lc >= 0 else "#ff1744"
    st.markdown(f'<div style="background:#1e2330; padding:20px; border-radius:12px; display:flex; align-items:center; gap:20px;"><div><div style="color:#888;">LIVE PRICE</div><div style="font-size:42px; font-weight:bold; color:{clr};">₹{lp:,.2f}</div></div><div style="background:{clr}15; color:{clr}; padding:5px 15px; border-radius:15px; font-weight:bold;">{lc:+.2f} ({lpct:+.2f}%)</div></div>', unsafe_allow_html=True)
    st.write("")
    
    if st.button("🚀 Run Multi-Factor AI Analysis"):
        with st.spinner("Fetching Data & Training AI with Indicators..."):
            try:
                # 1. Fetch Data
                p_map = {"1 Day": "1d", "1 Week": "5d", "1 Month": "1mo", "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"}
                period = p_map[chart_range]
                interval = "5m" if period in ["1d", "5d"] else "1d"
                df = yf.download(selected_ticker, period=period, interval=interval, progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                
                if not df.empty and len(df) > 60:
                    # 2. Add Indicators (NOW USED BY AI)
                    df = add_technical_indicators(df)
                    
                    # 3. Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='cyan', width=1), name="EMA 20"))
                    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 4. ADVANCED AI TRAINING
                    st.subheader("🧠 Multi-Factor AI Prediction")
                    st.caption("Analyzing Price + RSI + SMA + EMA simultaneously")
                    
                    # Features: Close, RSI, SMA_50, EMA_20 (4 Features)
                    data_features = df[['Close', 'RSI', 'SMA_50', 'EMA_20']].values
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(data_features)
                    
                    X_train, y_train = [], []
                    for i in range(60, len(scaled_data)):
                        X_train.append(scaled_data[i-60:i]) # Shape: (60, 4)
                        y_train.append(scaled_data[i, 0])   # Predict 'Close' (Index 0)
                    
                    X_train, y_train = np.array(X_train), np.array(y_train)
                    
                    # LSTM Model (Multi-Feature)
                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                    model.add(Dropout(0.2)) # Prevents overfitting
                    model.add(LSTM(50, return_sequences=False))
                    model.add(Dropout(0.2))
                    model.add(Dense(25))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0) # Increased epochs
                    
                    # Prediction
                    last_60 = scaled_data[-60:].reshape(1, 60, 4) # Must match 4 features
                    pred_scaled = model.predict(last_60)
                    
                    # Inverse Transform (Trickier with multi-feature)
                    # We create a dummy array to inverse transform correctly
                    dummy_array = np.zeros((1, 4))
                    dummy_array[0, 0] = pred_scaled[0, 0]
                    pred_price = scaler.inverse_transform(dummy_array)[0, 0]
                    
                    diff = pred_price - lp
                    color = "green" if diff > 0 else "red"
                    signal = "STRONG BUY 🚀" if diff > 0 else "STRONG SELL 🔻"
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("AI Target Price", f"₹{pred_price:.2f}")
                    c2.metric("Signal", signal)
                    c3.metric("Potential Move", f"{diff:+.2f}")
                    
                    # RSI Meter
                    last_rsi = df['RSI'].iloc[-1]
                    rsi_state = "Overbought (Risk)" if last_rsi > 70 else "Oversold (Buy Opp)" if last_rsi < 30 else "Neutral"
                    st.progress(int(last_rsi))
                    st.caption(f"RSI Momentum: {last_rsi:.2f} ({rsi_state})")

                    # 5. NEWS
                    st.markdown("---")
                    st.subheader(f"📰 {selected_ticker} News")
                    news = get_news_with_sources(selected_ticker.replace(".NS","").replace(".BO",""))
                    if news:
                        for n in news:
                            st.markdown(f'<div class="news-card"><div style="display:flex; justify-content:space-between; font-size:12px; color:#aaa;"><span>{n["source"]}</span><span>{n["date_str"]}</span></div><a href="{n["link"]}" target="_blank" style="color:white; font-weight:bold;">{n["title"]}</a><div style="font-size:12px; margin-top:5px;">{n["sentiment"]}</div></div>', unsafe_allow_html=True)
                    else:
                        st.info("No recent news found.")
                else:
                    st.error("Not enough data. Try a longer timeframe (at least 6 Months).")
            except Exception as e:
                st.error(f"Analysis Failed: {e}")
