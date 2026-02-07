import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
    page_title="Groww-Style Pro Dashboard", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# --------------------------
# 2. SESSION STATE MANAGEMENT
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Market Dashboard"
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = "RELIANCE.NS"

def navigate_to(page_name, ticker=None):
    st.session_state.page = page_name
    if ticker:
        st.session_state.selected_ticker = ticker
    st.rerun()

# --------------------------
# 3. MODERN CSS STYLING
st.markdown("""
<style>
    /* Global Settings */
    .stApp { background-color: #0f1115; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }
    
    /* GLASSMORPHISM CARDS */
    .fun-card {
        background: rgba(30, 34, 45, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .fun-card:hover { transform: translateY(-3px); border-color: #00d09c; }

    /* TEXT STYLES */
    .card-title { font-size: 13px; color: #aaa; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .card-price { font-size: 24px; font-weight: 700; color: #fff; margin: 5px 0; }
    .pos-change { color: #00d09c; font-weight: 600; font-size: 14px; }
    .neg-change { color: #ff4b4b; font-weight: 600; font-size: 14px; }
    
    /* NEWS */
    .news-box { padding: 12px; border-radius: 12px; background: #161920; margin-bottom: 10px; border-left: 3px solid #4c8bf5; }
    .news-link { color: #fff; text-decoration: none; font-weight: 500; font-size: 14px; }
    .news-link:hover { color: #4c8bf5; }
    
    /* BUTTON OVERRIDES */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        background-color: #1E1E1E;
        color: white;
        border: 1px solid #333;
        font-size: 12px;
    }
    div.stButton > button:hover {
        border-color: #00d09c;
        color: #00d09c;
        background-color: #1E1E1E;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. EXPANDED DATA POOLS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["ZOMATO.NS", "YESBANK.NS", "IDEA.NS", "TATASTEEL.NS", "RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS", "INFY.NS", "ITC.NS", "TATAMOTORS.NS"]

# --- NEW: EXPANDED ETF LIST ---
ETF_LIST = {
    "Nifty BeES": "NIFTYBEES.NS",
    "Bank BeES": "BANKBEES.NS",
    "Silver BeES": "SILVERBEES.NS",
    "Gold BeES": "GOLDBEES.NS",
    "IT BeES": "ITBEES.NS",
    "Pharma BeES": "PHARMABEES.NS",
    "PSU Bank BeES": "PSUBNKBEES.NS",
    "Auto BeES": "AUTOBEES.NS"
}

# --- NEW: EXPANDED COMMODITIES ---
COMMODITY_LIST = {
    "Gold (Global)": "GC=F",
    "Silver (Global)": "SI=F",
    "Crude Oil": "CL=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F"
}

RSS_FEEDS = {
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "CNBC-TV18": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/stock-market.xml",
    "LiveMint": "https://www.livemint.com/rss/markets"
}

# --------------------------
# 5. SIDEBAR
st.sidebar.title("🚀 Menu")
nav_options = ["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O & Options Chain", "🏦 ETFs & Mutual Funds", "🛢️ Global Commodities", "⭐ Top 5 AI Picks"]
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
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.fillna(0)

def train_ai(df):
    tf.keras.backend.clear_session()
    if len(df) < 60: return None, None
    df_ai = df[['Close','RSI','SMA_50','EMA_20']].fillna(0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_ai)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0: return None, None
    model = Sequential([LSTM(30, return_sequences=False, input_shape=(X.shape[1], 4)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    pred_scaled = model.predict(scaled[-60:].reshape(1, 60, 4), verbose=0)
    pred_price = scaler.inverse_transform([[pred_scaled[0,0],0,0,0]])[0,0]
    return pred_price, df['ATR'].iloc[-1]

# --------------------------
# 8. SHARED CHART COMPONENT (Used for Stock, ETF, Commodities)
def render_chart_page(default_ticker, title):
    # Header & Back
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"): navigate_to("🏠 Market Dashboard")
    with c2:
        st.subheader(f"{title}")

    # Search Bar
    col_ex, col_search = st.columns([1, 4])
    with col_ex:
        exchange = st.selectbox("Market", ["NSE", "BSE", "Global"], index=0 if "NS" in default_ticker else 2)
    with col_search:
        clean_default = default_ticker.replace(".NS","").replace(".BO","")
        query = st.text_input("Search Ticker", clean_default)

    if exchange == "NSE": suffix = ".NS"
    elif exchange == "BSE": suffix = ".BO"
    else: suffix = ""
    
    if query != clean_default:
        ticker_symbol = f"{query.upper().strip()}{suffix}"
    else:
        ticker_symbol = default_ticker

    # Fetch Data
    try:
        df_full = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
        
        # Flatten MultiIndex columns if present (Crucial Fix)
        if isinstance(df_full.columns, pd.MultiIndex):
            df_full.columns = df_full.columns.get_level_values(0)

        if df_full.empty:
            st.error(f"❌ No data found for {ticker_symbol}")
        else:
            df_full = add_indicators(df_full)
            curr = df_full['Close'].iloc[-1]
            diff = curr - df_full['Close'].iloc[-2]
            pct = (diff / df_full['Close'].iloc[-2]) * 100
            color = "#00e676" if diff >= 0 else "#ff1744"

            # Live Header
            st.markdown(f"""
            <div style="display:flex; align-items:baseline; gap:15px; margin-bottom:15px;">
                <h1 style="margin:0;">{query.upper()}</h1>
                <h2 style="margin:0; color:{color};">₹{curr:,.2f}</h2>
                <span style="color:{color}; font-weight:bold; font-size:18px;">{diff:+.2f} ({pct:+.2f}%)</span>
            </div>""", unsafe_allow_html=True)

            # Timeframe
            tf_map = {"1M":30,"6M":180,"1Y":365,"MAX":5000}
            sel_tf = st.radio("Range", list(tf_map.keys()), horizontal=True, label_visibility="collapsed")
            df_chart = df_full.tail(tf_map[sel_tf])

            # Tabs
            tab1, tab2 = st.tabs(["📊 Technicals & AI", "🧠 Fundamentals & News"])
            
            with tab1:
                # Chart
                fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'])])
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # AI Prediction
                st.caption("⚡ Analyzing patterns...")
                pred, atr = train_ai(df_full)
                if pred:
                    signal = "BUY 🚀" if pred > curr else "SELL 🔻"
                    col_ai1, col_ai2 = st.columns(2)
                    with col_ai1:
                         st.info(f"AI Target Price: ₹{pred:.2f}")
                    with col_ai2:
                         st.success(f"Signal: {signal}")

            with tab2:
                # Fundamentals
                try:
                    info = yf.Ticker(ticker_symbol).info
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Cap", f"{info.get('marketCap',0)/10000000:.2f} Cr")
                    c2.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                    c3.metric("Sector", info.get('sector', 'N/A'))
                except: st.info("Fundamental data unavailable.")
                
                # StockTwits Widget (Improved)
                clean_sym = query.upper().replace(".NS","")
                components.html(f"""
                <script type="text/javascript" src="https://api.stocktwits.com/addon/widget/2/widget-loader.min.js"></script>
                <div id="stocktwits-widget-news"></div>
                <script type="text/javascript">
                STWT.Widget({{
                    container: 'stocktwits-widget-news',
                    symbol: '{clean_sym}',
                    width: '100%',
                    height: '400',
                    limit: '15',
                    scrollbars: 'true',
                    streaming: 'true',
                    title: '{clean_sym} Live',
                    style: {{
                        link_color: '48515c',
                        link_hover_color: '48515c',
                        header_text_color: 'ffffff',
                        border_color: '333333',
                        divider_color: '333333',
                        box_color: '161920',
                        stream_color: '161920',
                        text_color: 'ffffff',
                        time_color: '999999'
                    }}
                }});
                </script>
                """, height=420, scrolling=True)

    except Exception as e: st.error(str(e))

# --------------------------
# 9. DASHBOARD VIEW
if view == "🏠 Market Dashboard":
    st.markdown('<div class="section-title">📊 Market Indices</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    for (name, ticker), col in zip(INDICES.items(), [c1, c2, c3]):
        p, ch, pct = get_live_data(ticker)
        color_cls = "pos-change" if ch >= 0 else "neg-change"
        sign = "+" if ch >= 0 else ""
        with col:
            st.markdown(f"""
            <div class="fun-card">
                <div class="card-title">{name}</div>
                <div class="card-price">₹{p:,.2f}</div>
                <div class="{color_cls}">{sign}{ch:.2f} ({sign}{pct:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Analyze {name} ➜", key=f"btn_{name}"):
                navigate_to("📈 Stock Analyzer", ticker)

    st.markdown('<div class="section-title">🔥 Most Traded (Live)</div>', unsafe_allow_html=True)
    top_vol_df = scan_most_traded()
    
    if not top_vol_df.empty:
        cols = st.columns(4)
        for i, (index, row) in enumerate(top_vol_df.iterrows()):
            sym = row['symbol'].replace(".NS","")
            pct = row['pct']
            color_cls = "pos-change" if pct >= 0 else "neg-change"
            sign = "+" if pct >= 0 else ""
            
            with cols[i]:
                st.markdown(f"""
                <div class="fun-card">
                    <div style="font-weight:700; font-size:16px;">{sym}</div>
                    <div style="font-size:18px; font-weight:bold; margin-top:5px;">₹{row['price']:.2f}</div>
                    <div class="{color_cls}">{sign}{pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Analyze {sym} ➜", key=f"btn_{sym}"):
                    navigate_to("📈 Stock Analyzer", row['symbol'])

    st.markdown("---")
    st.markdown('<div class="section-title">📰 Latest News</div>', unsafe_allow_html=True)
    news = get_news_multi()
    for n in news:
        st.markdown(f"<a href='{n['link']}' target='_blank' style='text-decoration:none; color:white;'><div class='news-box'><b>{n['source']}</b>: {n['title']} <span style='color:#888; font-size:11px'>({n['time']})</span></div></a>", unsafe_allow_html=True)

# --------------------------
# 10. STOCK ANALYZER
elif view == "📈 Stock Analyzer":
    render_chart_page(st.session_state.selected_ticker, "Stock Analysis")

# --------------------------
# 11. ETF PAGE (Now Functional)
elif view == "🏦 ETFs & Mutual Funds":
    st.markdown('<div class="section-title">🏦 Top ETFs</div>', unsafe_allow_html=True)
    
    # Selection Grid
    cols = st.columns(4) # Grid layout for many ETFs
    for i, (name, ticker) in enumerate(ETF_LIST.items()):
        with cols[i % 4]:
            if st.button(name, key=f"etf_{name}"):
                st.session_state.selected_ticker = ticker
    
    render_chart_page(st.session_state.selected_ticker, "ETF Analysis")

# --------------------------
# 12. COMMODITIES PAGE (Now Functional)
elif view == "🛢️ Global Commodities":
    st.markdown('<div class="section-title">🛢️ Global Commodities</div>', unsafe_allow_html=True)
    
    cols = st.columns(5)
    for i, (name, ticker) in enumerate(COMMODITY_LIST.items()):
        with cols[i]:
            if st.button(name, key=f"comm_{name}"):
                st.session_state.selected_ticker = ticker
                
    render_chart_page(st.session_state.selected_ticker, "Commodity Analysis")

# --------------------------
# 13. F/O DASHBOARD
elif view == "📊 F/O & Options Chain":
    st.markdown('<div class="section-title">📊 F/O Dashboard</div>', unsafe_allow_html=True)
    fo_ticker = st.text_input("Enter Symbol (e.g. NIFTY, BANKNIFTY, RELIANCE)", "NIFTY")
    
    if fo_ticker.upper() == "NIFTY": yf_sym = "^NSEI"
    elif fo_ticker.upper() == "BANKNIFTY": yf_sym = "^NSEBANK"
    else: yf_sym = f"{fo_ticker.upper()}.NS"

    try:
        tk = yf.Ticker(yf_sym)
        spot = tk.fast_info.last_price
        st.metric("Spot Price", f"₹{spot:,.2f}")
        
        # Options Data
        expiries = tk.options
        if expiries:
            sel_exp = st.selectbox("Expiry Date", expiries)
            chain = tk.option_chain(sel_exp)
            calls = chain.calls
            puts = chain.puts
            
            # PCR
            pcr = puts['openInterest'].sum() / calls['openInterest'].sum()
            st.info(f"Put-Call Ratio (PCR): {pcr:.2f}")

            # ATM Filter
            strike_min = spot * 0.98
            strike_max = spot * 1.02
            
            c_atm = calls[(calls['strike']>=strike_min) & (calls['strike']<=strike_max)][['strike','lastPrice','openInterest']]
            p_atm = puts[(puts['strike']>=strike_min) & (puts['strike']<=strike_max)][['strike','lastPrice','openInterest']]
            
            c_atm.rename(columns={'lastPrice':'CE Price', 'openInterest':'CE OI'}, inplace=True)
            p_atm.rename(columns={'lastPrice':'PE Price', 'openInterest':'PE OI'}, inplace=True)
            
            df_chain = pd.merge(c_atm, p_atm, on='strike')
            st.dataframe(df_chain.set_index('strike'), use_container_width=True)
        else:
            st.warning("Options data not available for this symbol.")
            
    except Exception as e:
        st.error(f"Error loading F/O data: {e}")

# --------------------------
# 14. AI PICKS
elif view == "⭐ Top 5 AI Picks":
    st.markdown('<div class="section-title">⭐ AI Market Scanners</div>', unsafe_allow_html=True)
    st.info("Running AI scan on market leaders...")
    
    # Simplified Logic for Demo Speed
    results = []
    progress = st.progress(0)
    for i, ticker in enumerate(SCANNER_POOL[:8]): # Limit to 8 for speed
        progress.progress((i+1)/8)
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = add_indicators(df)
            pred, _ = train_ai(df)
            if pred:
                curr = df['Close'].iloc[-1]
                diff = pred - curr
                results.append((ticker, diff, curr))
        except: continue
    
    progress.empty()
    
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        for sym, diff, curr in results:
            action = "BUY 🚀" if diff > 0 else "SELL 🔻"
            color = "#00d09c" if diff > 0 else "#ff4b4b"
            st.markdown(f"""
            <div class="fun-card" style="border-left: 5px solid {color}">
                <h3>{sym.replace('.NS','')}</h3>
                <p>Current: ₹{curr:.2f}</p>
                <p style="color:{color}; font-weight:bold;">Potential: {diff:+.2f} ({action})</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No clear signals found right now.")
