import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
from datetime import datetime, timedelta
import pytz
import feedparser
from textblob import TextBlob
import time
import requests
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# --------------------------
# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Groww-Style Dashboard", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# --------------------------
# 2. SESSION STATE
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Market Dashboard"

def navigate_to(page_name):
    st.session_state.page = page_name

# --------------------------
# 3. CSS STYLING
st.markdown("""
<style>
    .stApp { background-color: #0f1115; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }
    
    /* CARDS */
    .fun-card {
        background: rgba(30, 34, 45, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fun-card:hover { transform: translateY(-5px); border-color: #00d09c; }

    /* TEXT GRADIENTS */
    .gradient-text-green { background: -webkit-linear-gradient(45deg, #00d09c, #00ffaa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .gradient-text-red { background: -webkit-linear-gradient(45deg, #ff4b4b, #ff9068); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    
    /* HEADERS */
    .section-title { font-size: 22px; font-weight: 700; margin-top: 25px; margin-bottom: 15px; color: #e0e0e0; }

    /* METRICS */
    .metric-label { font-size: 12px; color: #888; }
    .metric-val { font-size: 16px; font-weight: 600; color: #fff; }

    /* NEWS */
    .news-box { padding: 12px; border-radius: 12px; background: #161920; margin-bottom: 10px; border-left: 3px solid #4c8bf5; }
    .news-link { color: #fff; text-decoration: none; font-weight: 500; font-size: 14px; }
    .news-link:hover { color: #4c8bf5; }
    .news-meta { font-size: 11px; color: #888; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. DATA POOLS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["ZOMATO.NS", "YESBANK.NS", "IDEA.NS", "TATASTEEL.NS", "RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS", "INFY.NS", "ITC.NS", "TATAMOTORS.NS"]
RSS_FEEDS = {
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "CNBC-TV18": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/stock-market.xml",
    "LiveMint": "https://www.livemint.com/rss/markets"
}

# --------------------------
# 5. SIDEBAR
st.sidebar.title("🚀 Menu")
nav_options = ["🏠 Market Dashboard", "📈 Stock Analyzer", "🏦 ETFs & Mutual Funds", "🛢️ Global Commodities", "⭐ Top 5 AI Picks"]
selection = st.sidebar.radio("Navigate", nav_options, index=nav_options.index(st.session_state.page))
if selection != st.session_state.page:
    st.session_state.page = selection
    st.rerun()
view = st.session_state.page

# --------------------------
# 6. AUTO REFRESH
if view != "⭐ Top 5 AI Picks":
    st_autorefresh(interval=30000, key="dashboard_refresh")

# --------------------------
# 7. UTILITIES
def get_live_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.fast_info
        price = info.last_price
        prev = info.previous_close
        if price is None or prev is None: return 0.0, 0.0, 0.0
        return price, price - prev, (price - prev) / prev * 100
    except:
        return 0.0, 0.0, 0.0

@st.cache_data(ttl=60)
def scan_most_traded():
    data = []
    for ticker in SCANNER_POOL:
        try:
            t = yf.Ticker(ticker)
            vol = t.fast_info.last_volume
            price = t.fast_info.last_price
            prev = t.fast_info.previous_close
            pct = ((price-prev)/prev)*100
            data.append({"symbol": ticker, "vol": vol, "price": price, "pct": pct})
        except: continue
    df = pd.DataFrame(data)
    if not df.empty:
        return df.sort_values(by='vol', ascending=False).head(4)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_news_multi():
    items = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:2]:
                ts = time.mktime(e.published_parsed) if 'published_parsed' in e else time.time()
                date_str = time.strftime('%H:%M', time.localtime(ts))
                items.append({'title':e.title, 'link':e.link, 'source':source, 'ts':ts, 'time':date_str})
        except: continue
    return sorted(items, key=lambda x: x['ts'], reverse=True)[:6]

def add_indicators(df):
    if df.empty: return df
    df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, min_periods=1).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.fillna(0)

# AI Training Function
def train_ai(df):
    tf.keras.backend.clear_session()
    # Ensure enough data points
    if len(df) < 60: return None, None

    # Prepare Data
    df_ai = df[['Close','RSI','SMA_50','EMA_20']].fillna(0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_ai)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    if len(X) == 0: return None, None

    # Train Simplified Model
    model = Sequential([
        LSTM(30, return_sequences=False, input_shape=(X.shape[1], 4)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=3, batch_size=16, verbose=0)

    # Predict Next Candle
    last60 = scaled[-60:].reshape(1, 60, 4)
    pred_scaled = model.predict(last60, verbose=0)
    dummy = np.zeros((1, 4))
    dummy[0, 0] = pred_scaled[0, 0]
    pred_price = scaler.inverse_transform(dummy)[0, 0]

    return pred_price, df['ATR'].iloc[-1]

# --------------------------
# 8. DASHBOARD VIEW
if view == "🏠 Market Dashboard":
    st.markdown('<div class="section-title">📊 Market Indices</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for (name, ticker), col in zip(INDICES.items(), [c1, c2, c3]):
        p, ch, pct = get_live_data(ticker)
        color = "#00d09c" if ch >= 0 else "#ff4b4b"
        icon = "🟢" if ch >= 0 else "🔴"
        with col:
            st.markdown(f"""
            <div class="fun-card" style="border-top: 3px solid {color};">
                <div style="font-size:12px; color:#888; font-weight:600;">{icon} {name}</div>
                <div style="font-size:24px; font-weight:700; color:#fff;">{p:,.2f}</div>
                <div style="color:{color}; font-weight:600; font-size:14px;">{ch:+.2f} ({pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🛠️ Products & Tools</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    def draw_tool(icon, name, page, col):
        with col:
            if st.button(f"{icon}\n{name}", key=name, use_container_width=True):
                navigate_to(page)
                st.rerun()
    draw_tool("📈", "Stocks", "📈 Stock Analyzer", c1)
    draw_tool("🏦", "ETFs", "🏦 ETFs & Mutual Funds", c2)
    draw_tool("🤖", "AI Picks", "⭐ Top 5 AI Picks", c3)
    draw_tool("🛢️", "Commodities", "🛢️ Global Commodities", c4)

    st.markdown('<div class="section-title">🔥 Most Traded (Live)</div>', unsafe_allow_html=True)
    top_vol_df = scan_most_traded()
    if not top_vol_df.empty:
        cols = st.columns(4)
        for i, (index, row) in enumerate(top_vol_df.iterrows()):
            sym = row['symbol'].replace(".NS","")
            pct = row['pct']
            color_class = "gradient-text-green" if pct >= 0 else "gradient-text-red"
            sign = "+" if pct >= 0 else ""
            with cols[i]:
                st.markdown(f"""
                <div class="fun-card" style="text-align:center;">
                    <img src="https://img.icons8.com/fluency/48/bullish.png" width="35" style="margin-bottom:8px; opacity:0.8;">
                    <div style="font-weight:700; font-size:15px; margin-bottom:4px;">{sym}</div>
                    <div style="font-size:18px; font-weight:bold;">₹{row['price']:.2f}</div>
                    <div class="{color_class}" style="font-size:14px;">{sign}{pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📰 Market Updates</div>', unsafe_allow_html=True)
    news = get_news_multi()
    news_cols = st.columns(2)
    for i, n in enumerate(news):
        with news_cols[i % 2]:
            st.markdown(f"""
            <div class="news-box">
                <div class="news-meta">{n['source']} • {n['time']}</div>
                <a href="{n['link']}" target="_blank" class="news-link">{n['title']}</a>
            </div>
            """, unsafe_allow_html=True)

# --------------------------
# 9. STOCK ANALYZER (UPDATED)
elif view == "📈 Stock Analyzer":
    
    # 1. SEARCH BAR
    col_ex, col_search = st.columns([1, 4])
    with col_ex:
        exchange = st.selectbox("Exch", ["NSE", "BSE"])
    with col_search:
        query = st.text_input("Search Stock (e.g. ZOMATO, RELIANCE)", "RELIANCE")

    suffix = ".NS" if exchange == "NSE" else ".BO"
    ticker_symbol = f"{query.upper().strip()}{suffix}"

    # 2. FETCH DATA (Handling MultiIndex Issue)
    try:
        # Download lots of data for AI (2y), but we slice for chart later
        df_full = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)

        # FIX: Flatten columns if they are MultiIndex (The Blank Graph Fix)
        if isinstance(df_full.columns, pd.MultiIndex):
            df_full.columns = df_full.columns.get_level_values(0)

        if df_full.empty:
            st.error(f"❌ Could not fetch data for **{ticker_symbol}**. Please check symbol.")
        else:
            df_full = add_indicators(df_full)
            curr_price = df_full['Close'].iloc[-1]
            prev_close = df_full['Close'].iloc[-2]
            change = curr_price - prev_close
            pct_change = (change / prev_close) * 100
            color = "#00e676" if change >= 0 else "#ff1744"

            # 3. LIVE HEADER
            st.markdown(f"""
            <div style="display:flex; align-items:baseline; gap:15px; margin-bottom:15px;">
                <h1 style="margin:0; font-size:32px;">{query.upper()}</h1>
                <h2 style="margin:0; color:{color};">₹{curr_price:,.2f}</h2>
                <span style="color:{color}; font-weight:bold; font-size:18px;">{change:+.2f} ({pct_change:+.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)

            # 4. TIMEFRAME SELECTOR (For Chart)
            tf_cols = st.columns(8)
            timeframes = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "MAX": 5000}
            selected_tf = st.radio("Range", list(timeframes.keys()), horizontal=True, label_visibility="collapsed")
            
            # Slice dataframe for chart
            days = timeframes[selected_tf]
            df_chart = df_full.tail(days)

            # 5. TABS
            tab1, tab2 = st.tabs(["📊 Technical & AI", "🧠 Fundamentals & Sentiment"])

            with tab1:
                # CHART
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # AUTOMATIC AI PREDICTION
                st.subheader("🤖 AI Trade Signal")
                with st.spinner("Running AI Model..."):
                    pred, atr = train_ai(df_full) # Use full data for AI
                
                if pred:
                    diff = pred - curr_price
                    sig = "BUY 🚀" if diff > 0 else "SELL 🔻"
                    card_color = "#00c853" if diff > 0 else "#d50000"
                    
                    sl = curr_price - (1.5 * atr) if diff > 0 else curr_price + (1.5 * atr)
                    tgt = curr_price + (2.0 * atr) if diff > 0 else curr_price - (2.0 * atr)

                    st.markdown(f"""
                    <div style="background:{card_color}20; border-left:5px solid {card_color}; padding:15px; border-radius:10px;">
                        <div style="font-size:18px; font-weight:bold; color:white;">Signal: {sig}</div>
                        <div style="display:flex; gap:20px; margin-top:10px;">
                            <div>🎯 Target: <span style="font-weight:bold; color:#fff;">{tgt:.2f}</span></div>
                            <div>🛑 StopLoss: <span style="font-weight:bold; color:#fff;">{sl:.2f}</span></div>
                            <div>🔮 AI Pred: <span style="font-weight:bold; color:#fff;">{pred:.2f}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Not enough data for AI prediction.")

            with tab2:
                # FUNDAMENTALS
                st.subheader("📊 Key Fundamentals")
                try:
                    info = yf.Ticker(ticker_symbol).info
                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("Market Cap", f"₹{info.get('marketCap', 0)/10000000:.0f} Cr")
                    f2.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
                    f3.metric("52W High", f"{info.get('fiftyTwoWeekHigh', 0)}")
                    f4.metric("Sector", info.get('sector', 'N/A'))
                except:
                    st.info("Fundamental data unavailable via API.")

                st.markdown("---")
                st.subheader("🗣️ Social Sentiment (StockTwits)")
                
                # External Links Backup
                clean_sym = query.upper()
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"""<a href="https://www.screener.in/company/{clean_sym}/" target="_blank" class="ext-btn">📊 Screener.in</a>""", unsafe_allow_html=True)
                c2.markdown(f"""<a href="https://stockedge.com/search?query={clean_sym}" target="_blank" class="ext-btn">📈 StockEdge</a>""", unsafe_allow_html=True)
                c3.markdown(f"""<a href="https://www.tradingview.com/chart/?symbol={clean_sym}" target="_blank" class="ext-btn">📉 TradingView</a>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --------------------------
# 10. OTHER PLACEHOLDERS
else:
    st.markdown(f'<div class="section-title">{view}</div>', unsafe_allow_html=True)
    if st.button("← Back to Dashboard"):
        navigate_to("🏠 Market Dashboard")
        st.rerun()
    st.info("Feature under maintenance.")
