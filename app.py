import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from textblob import TextBlob
import feedparser
import urllib.parse
import datetime

# --- 1. APP CONFIGURATION & STYLING ---
st.set_page_config(page_title="Super Stock AI (Pro)", layout="wide", page_icon="📈")

# Custom CSS for "TradingView-like" Look
st.markdown("""
<style>
    /* Main Background - Dark Blue/Black Gradient */
    .stApp {
        background: linear-gradient(to bottom right, #0e1117, #151922);
        color: #ffffff;
    }
    
    /* Custom Card Styling */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #1e2330;
        border: 1px solid #2a2f3d;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00e676; /* Bright Green */
        font-weight: 700;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #12151e;
        border-right: 1px solid #2a2f3d;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #2962ff, #0039cb);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(41, 98, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ASSET DATABASE (The "Smart List") ---
# This mimics the auto-complete dropdown
ASSETS = {
    "Stocks (India)": {
        "Reliance Industries": "RELIANCE",
        "TCS": "TCS",
        "HDFC Bank": "HDFCBANK",
        "Infosys": "INFY",
        "ICICI Bank": "ICICIBANK",
        "Tata Motors": "TATAMOTORS",
        "State Bank of India": "SBIN",
        "Adani Enterprises": "ADANIENT",
        "Bajaj Finance": "BAJFINANCE",
        "Wipro": "WIPRO",
        "Zomato": "ZOMATO",
        "Paytm": "PAYTM",
        "Suzlon Energy": "SUZLON"
    },
    "Commodities (MCX/Global)": {
        "Gold (Global Spot)": "GC=F",
        "Silver (Global Spot)": "SI=F",
        "Crude Oil (WTI)": "CL=F",
        "Gold Bees (India ETF)": "GOLDBEES",
        "Silver Bees (India ETF)": "SILVERBEES"
    },
    "Indices (F/O)": {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "Sensex": "^BSESN",
        "India VIX": "^INDIAVIX"
    },
    "Crypto": {
        "Bitcoin (USD)": "BTC-USD",
        "Ethereum (USD)": "ETH-USD",
        "Solana (USD)": "SOL-USD"
    }
}

# --- 3. HELPER FUNCTIONS (The "Brain") ---

def fix_timezone(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Asia/Kolkata')
    return df

def get_real_time_price(symbol):
    try:
        live_data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if not live_data.empty:
            if live_data.index.tz is None:
                live_data.index = live_data.index.tz_localize('UTC')
            live_data.index = live_data.index.tz_convert('Asia/Kolkata')
            return live_data['Close'].iloc[-1].item(), live_data.index[-1]
        return None, None
    except:
        return None, None

def get_data_params(predict_target, view_target):
    if predict_target == "Next 5 Minutes": interval = "5m"; period = "5d"
    elif predict_target == "Next 15 Minutes": interval = "15m"; period = "5d"
    elif predict_target == "Next 30 Minutes": interval = "30m"; period = "1mo"
    elif predict_target == "Next 1 Hour": interval = "60m"; period = "3mo"
    else: interval = "1d"; period = "5y"
    
    view_map = {
        "1 Day": "1d", "5 Days": "5d", "1 Month": "1mo", 
        "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"
    }
    return interval, period, view_map[view_target]

def add_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    return df

def get_news_sentiment(ticker_symbol):
    try:
        query = urllib.parse.quote(ticker_symbol.replace(".NS","").replace(".BO",""))
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        news_data = []
        if feed.entries:
            for entry in feed.entries[:5]: 
                analysis = TextBlob(entry.title)
                polarity = analysis.sentiment.polarity
                sentiment = "🟢 Positive" if polarity > 0.05 else "🔴 Negative" if polarity < -0.05 else "⚪ Neutral"
                news_data.append({'Title': entry.title, 'Link': entry.link, 'Sentiment': sentiment})
        return news_data
    except:
        return []

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.markdown("## ⚙️ Control Panel")

# A. Asset Selection
asset_category = st.sidebar.selectbox("📂 Asset Class", list(ASSETS.keys()))
selected_asset_name = st.sidebar.selectbox("🔍 Select Asset", list(ASSETS[asset_category].keys()))
base_ticker = ASSETS[asset_category][selected_asset_name]

# B. Exchange Selection (Only for Indian Stocks/ETFs)
final_ticker = base_ticker
if asset_category in ["Stocks (India)", "Commodities (MCX/Global)"]:
    # Only show toggle if it's an Indian asset (not global commodity futures like GC=F)
    if "=" not in base_ticker: 
        exchange = st.sidebar.radio("🏛️ Exchange", ["NSE", "BSE"], horizontal=True)
        suffix = ".NS" if exchange == "NSE" else ".BO"
        final_ticker = base_ticker + suffix

st.sidebar.markdown("---")
prediction_option = st.sidebar.selectbox("🎯 Prediction Target", ("Next 5 Minutes", "Next 15 Minutes", "Next 1 Hour", "Next 1 Day"))
chart_view = st.sidebar.selectbox("👀 Chart History", ("1 Day", "5 Days", "1 Month", "1 Year", "5 Years"))

# --- 5. MAIN APP LAYOUT ---

st.title(f"⚡ {selected_asset_name} ({final_ticker})")

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    
    interval, ai_period, view_period = get_data_params(prediction_option, chart_view)
    
    with st.spinner(f"📡 Connecting to Exchange... Fetching {final_ticker}"):
        try:
            # Fetch Data
            data = yf.download(final_ticker, period=ai_period, interval=interval, progress=False)
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
            
            real_price, real_time = get_real_time_price(final_ticker)
            
            if len(data) > 0:
                data = fix_timezone(data)
                
                # --- DASHBOARD METRICS ---
                display_price = real_price if real_price else data['Close'].iloc[-1]
                stock_fast = yf.Ticker(final_ticker)
                prev_close = stock_fast.fast_info.get('previous_close', display_price)
                
                change = display_price - prev_close
                pct_change = (change / prev_close) * 100
                color_hex = "#00e676" if change >= 0 else "#ff1744"
                
                # Custom HTML Metric Card
                st.markdown(f"""
                <div style="background-color: #1e2330; padding: 20px; border-radius: 12px; border-left: 5px solid {color_hex};">
                    <h3 style="margin:0; color: #b0bec5;">Live Price</h3>
                    <h1 style="margin:0; font-size: 42px; color: {color_hex};">₹{display_price:,.2f}</h1>
                    <h4 style="margin:0; color: {color_hex};">{change:+.2f} ({pct_change:+.2f}%)</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### ") # Spacer

                # --- CHART ---
                data = add_indicators(data)
                if chart_view == "1 Day": chart_data = data.tail(75)
                elif chart_view == "5 Days": chart_data = data.tail(375)
                else: chart_data = data
                
                fig = go.Figure()
                date_fmt = '%H:%M' if interval != '1d' else '%d-%b'
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=chart_data.index.strftime(date_fmt),
                    open=chart_data['Open'], high=chart_data['High'],
                    low=chart_data['Low'], close=chart_data['Close'], name='Price'
                ))
                # SMA
                fig.add_trace(go.Scatter(
                    x=chart_data.index.strftime(date_fmt), 
                    y=chart_data['SMA_50'], line=dict(color='#2962ff', width=2), name='SMA 50'
                ))
                
                fig.update_layout(
                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False, type='category'),
                    yaxis=dict(showgrid=True, gridcolor='#2a2f3d'),
                    height=500, margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- AI PREDICTION ---
                if len(data) > 60:
                    st.markdown("### 🤖 Neural Network Prediction")
                    progress_bar = st.progress(0)
                    
                    # Data Prep
                    df_ai = data[['Close']].values
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(df_ai)
                    
                    x_train, y_train = [], []
                    for i in range(60, len(scaled_data)):
                        x_train.append(scaled_data[i-60:i, 0])
                        y_train.append(scaled_data[i, 0])
                    
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    
                    progress_bar.progress(50)
                    
                    # LSTM Model
                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(LSTM(50, return_sequences=False))
                    model.add(Dense(25))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(x_train, y_train, batch_size=1, epochs=3, verbose=0)
                    
                    progress_bar.progress(90)
                    
                    # Predict
                    last_60 = scaled_data[-60:].reshape(1, 60, 1)
                    pred_price = scaler.inverse_transform(model.predict(last_60))[0][0]
                    
                    progress_bar.progress(100)
                    
                    # Result Card
                    diff = pred_price - display_price
                    sig_color = "#00e676" if diff > 0 else "#ff1744"
                    direction = "BULLISH 🚀" if diff > 0 else "BEARISH 🔻"
                    
                    c1, c2 = st.columns(2)
                    c1.metric(f"AI Target ({prediction_option})", f"₹{pred_price:.2f}")
                    c2.markdown(f"""
                    <div style="background-color: {sig_color}20; padding: 10px; border-radius: 8px; border: 1px solid {sig_color}; text-align: center;">
                        <h3 style="margin:0; color: {sig_color};">{direction}</h3>
                        <p style="margin:0;">Potential: {diff:+.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # --- NEWS FEED ---
                st.markdown("---")
                st.subheader("📰 Market Sentiment")
                news = get_news_sentiment(final_ticker)
                
                if news:
                    for n in news:
                        st.markdown(f"**{n['Sentiment']}** [{n['Title']}]({n['Link']})")
                else:
                    st.info("No News Found")

            else:
                st.error("❌ Data Unavailable. Market might be closed or Ticker is invalid.")
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("👈 Select an asset from the sidebar and click 'Run Analysis' to start!")
