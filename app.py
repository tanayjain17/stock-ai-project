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
from datetime import datetime, time as dt_time
import time
import tensorflow as tf
import pytz # TIMEZONE SUPPORT

# --- 0. SEED SETTING ---
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Market Pulse AI", layout="wide", page_icon="⚡")

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
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
    }
    .market-open { background-color: #00e676; color: black; animation: pulse 1.5s infinite; }
    .market-closed { background-color: #ff1744; color: white; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
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

# --- 3. SMART HELPERS (The Safety Brain) ---

def is_market_open(ticker):
    # 1. If it's Crypto or Global, assume OPEN (24/7 or diff timezones)
    if not (ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^")):
        return True, "Global / 24x7"

    # 2. Setup Timezones
    utc_now = datetime.now(pytz.utc)
    ist_now = utc_now.astimezone(pytz.timezone('Asia/Kolkata'))
    
    # 3. Check Weekends (Saturday=5, Sunday=6)
    if ist_now.weekday() >= 5:
        return False, "Weekend Closed"
        
    # 4. Check NSE Hours (09:15 to 15:30)
    current_time = ist_now.time()
    market_start = dt_time(9, 15)
    market_end = dt_time(15, 30)
    
    if market_start <= current_time <= market_end:
        return True, "Market Open"
    else:
        return False, "Market Closed"

def get_currency_symbol(ticker):
    if ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^"): return "₹"
    else: return "$"

def get_live_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.fast_info.last_price
        prev_close = stock.fast_info.previous_close
        return price, price - prev_close, (price - prev_close) / prev_close * 100
    except:
        return 0.0, 0.0, 0.0

def add_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

def train_ai_model(df):
    data_features = df[['Close', 'RSI', 'SMA_50', 'EMA_20']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data_features)
    
    if len(scaled_data) < 60: return None, None, None
    
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i]) 
        y_train.append(scaled_data[i, 0])   
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2)) 
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0)
    
    last_60 = scaled_data[-60:].reshape(1, 60, 4)
    pred_scaled = model.predict(last_60)
    
    dummy_array = np.zeros((1, 4))
    dummy_array[0, 0] = pred_scaled[0, 0]
    pred_price = scaler.inverse_transform(dummy_array)[0, 0]
    
    return pred_price, model, scaler

# --- 4. SIDEBAR ---
st.sidebar.title("⚡ Flash Scanner")
view_mode = st.sidebar.radio("Navigation", ["🏠 Dashboard", "📈 Analyzer"])

selected_ticker = "RELIANCE.NS"
if view_mode == "📈 Analyzer":
    stock_name = st.sidebar.selectbox("Stocks", list(NIFTY_100_TICKERS.keys()))
    selected_ticker = NIFTY_100_TICKERS[stock_name]
    custom = st.sidebar.text_input("Ticker Search")
    if custom: selected_ticker = f"{custom.upper()}.NS"

# --- 5. PAGE LOGIC ---

# >>> SMART AUTO-REFRESH FRAGMENT <<<
@st.fragment(run_every=1) 
def show_live_price_and_chart(ticker):
    # 1. CHECK MARKET STATUS
    is_open, status_msg = is_market_open(ticker)
    curr_sym = get_currency_symbol(ticker)
    
    # 2. FETCH PRICE (Only if Open or first run)
    # We always fetch "fast_info" price because it's cheap and safe.
    # But we STOP the heavy chart download if closed.
    lp, lc, lpct = get_live_price(ticker)
    clr = "#00e676" if lc >= 0 else "#ff1744"
    
    # BADGE LOGIC
    badge_class = "market-open" if is_open else "market-closed"
    badge_text = f"LIVE ⚡" if is_open else f"CLOSED ({status_msg})"
    
    st.markdown(f"""
    <div style="background:#1e2330; padding:15px; border-radius:12px; display:flex; align-items:center; gap:20px; margin-bottom:15px;">
        <div>
            <div style="color:#888; font-size:11px; margin-bottom:4px;">MARKET STATUS <span class="status-badge {badge_class}">{badge_text}</span></div>
            <div style="font-size:38px; font-weight:bold; color:{clr};">{curr_sym}{lp:,.2f}</div>
        </div>
        <div style="background:{clr}15; color:{clr}; padding:5px 12px; border-radius:15px; font-weight:bold; font-size:14px;">
            {lc:+.2f} ({lpct:+.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. SMART CHART FETCH
    # If Market is CLOSED, we don't need to ping Yahoo every second for a chart that won't change.
    # We only fetch if it's OPEN. If closed, we show a static message or a cached view.
    
    if is_open:
        try:
            df_live = yf.download(ticker, period="1d", interval="1m", progress=False)
            if isinstance(df_live.columns, pd.MultiIndex): df_live.columns = df_live.columns.droplevel(1)
            
            if not df_live.empty:
                df_live = add_technical_indicators(df_live)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_live.index, open=df_live['Open'], high=df_live['High'], low=df_live['Low'], close=df_live['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df_live.index, y=df_live['EMA_20'], line=dict(color='cyan', width=1), name="EMA 20"))
                fig.update_layout(height=400, title=f"Real-Time 1m Chart", template="plotly_dark", margin=dict(t=30,b=0,l=0,r=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Connecting...")
    else:
        # MARKET CLOSED MODE
        st.info(f"✨ Data Feed Paused (Market Closed). Last Close Price: {lp}")
        # Optional: You could show a static 'Daily' chart here if you wanted, 
        # but to save resources we just show the info message.

# --- MAIN APP UI ---
if view_mode == "🏠 Dashboard":
    st.title("🌏 Market Dashboard")
    
    # We also gate the dashboard refresh. 
    # If it's night time, we don't need to refresh indices every 5s.
    @st.fragment(run_every=5)
    def show_index_dashboard():
        col1, col2, col3 = st.columns(3)
        for n, s, c in [("NIFTY 50", "^NSEI", col1), ("SENSEX", "^BSESN", col2), ("BANK NIFTY", "^NSEBANK", col3)]:
            # Check if Index is open (Same logic as stocks)
            open_status, _ = is_market_open(s)
            
            # If closed, we can just show static data (but fast_info is cheap, so we keep it active for indices)
            p, ch, pc = get_live_price(s)
            clr = "#00e676" if ch >= 0 else "#ff1744"
            curr = get_currency_symbol(s)
            with c:
                st.markdown(f'<div class="metric-card" style="border-top: 3px solid {clr};"><div>{n}</div><div style="font-size:24px; font-weight:bold;">{curr}{p:,.2f}</div><div style="color:{clr};">{ch:+.2f} ({pc:+.2f}%)</div></div>', unsafe_allow_html=True)
    show_index_dashboard()

else:
    st.title(f"⚡ {selected_ticker} Live Station")
    show_live_price_and_chart(selected_ticker)
    
    st.write("")
    if st.button("Run AI Prediction"):
        with st.spinner("Analyzing..."):
            try:
                df_train = yf.download(selected_ticker, period="1y", interval="1d", progress=False)
                if isinstance(df_train.columns, pd.MultiIndex): df_train.columns = df_train.columns.droplevel(1)
                
                if not df_train.empty:
                    df_train = add_technical_indicators(df_train)
                    pred_price, _, _ = train_ai_model(df_train)
                    
                    if pred_price:
                        curr_price, _, _ = get_live_price(selected_ticker)
                        diff = pred_price - curr_price
                        sig = "BUY 🚀" if diff > 0 else "SELL 🔻"
                        c_sym = get_currency_symbol(selected_ticker)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("AI Target", f"{c_sym}{pred_price:.2f}")
                        col2.metric("Signal", sig, f"{diff:+.2f}")
                    else:
                        st.error("Insufficient Data")
            except Exception as e:
                st.error("Analysis Failed")
