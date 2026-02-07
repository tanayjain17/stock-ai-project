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
    
    /* DASHBOARD CARD BUTTONS */
    /* This transforms standard Streamlit buttons into Dashboard Cards */
    div.stButton > button {
        width: 100%;
        height: auto;
        padding: 15px 20px;
        background: rgba(30, 34, 45, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        color: white;
        text-align: left;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    
    div.stButton > button:hover {
        transform: translateY(-4px);
        border-color: #00d09c;
        background: rgba(30, 34, 45, 0.8);
        box-shadow: 0 8px 16px rgba(0, 208, 156, 0.1);
    }

    div.stButton > button p {
        margin: 0;
        padding: 0;
    }

    /* NEWS BOX */
    .news-box { 
        padding: 15px; 
        border-radius: 12px; 
        background: #161920; 
        margin-bottom: 12px; 
        border-left: 4px solid #4c8bf5; 
        transition: 0.2s;
    }
    .news-box:hover { background: #1e222b; }
    .news-link { color: #e0e0e0; text-decoration: none; font-weight: 500; font-size: 14px; display: block; }
    .news-meta { font-size: 11px; color: #888; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }

    /* SIGNAL CARD */
    .signal-card {
        background: #1e222d;
        border-radius: 15px;
        padding: 20px;
        margin-top: 15px;
        border: 1px solid #333;
    }
    
    /* SECTION HEADERS */
    .section-title { 
        font-size: 20px; 
        font-weight: 700; 
        margin-top: 25px; 
        margin-bottom: 15px; 
        color: #f0f0f0; 
        display: flex; 
        align-items: center; 
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. DATA POOLS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["ZOMATO.NS", "YESBANK.NS", "IDEA.NS", "TATASTEEL.NS", "RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS", "INFY.NS", "ITC.NS", "TATAMOTORS.NS"]
ETF_LIST = {"Nifty BeES": "NIFTYBEES.NS", "Gold BeES": "GOLDBEES.NS", "Silver BeES": "SILVERBEES.NS", "Bank BeES": "BANKBEES.NS", "IT BeES": "ITBEES.NS"}
COMMODITY_LIST = {"Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F", "Natural Gas": "NG=F", "Copper": "HG=F"}

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
def get_currency_symbol(ticker):
    if ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^"):
        return "₹"
    return "$"

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
# 8. SHARED CHART COMPONENT
def render_chart_page(default_ticker, title):
    # Header & Back
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"): navigate_to("🏠 Market Dashboard")
    with c2:
        st.subheader(title)

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
    
    ticker_symbol = f"{query.upper().strip()}{suffix}" if query != clean_default else default_ticker
    curr_sym = get_currency_symbol(ticker_symbol)

    try:
        df_full = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
        if isinstance(df_full.columns, pd.MultiIndex): df_full.columns = df_full.columns.get_level_values(0)

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
                <h2 style="margin:0; color:{color};">{curr_sym}{curr:,.2f}</h2>
                <span style="color:{color}; font-weight:bold; font-size:18px;">{diff:+.2f} ({pct:+.2f}%)</span>
            </div>""", unsafe_allow_html=True)

            tf_map = {"1M":30,"6M":180,"1Y":365,"MAX":5000}
            sel_tf = st.radio("Range", list(tf_map.keys()), horizontal=True, label_visibility="collapsed")
            df_chart = df_full.tail(tf_map[sel_tf])

            tab1, tab2 = st.tabs(["📊 Technicals & AI", "🧠 Fundamentals & News"])
            
            with tab1:
                fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'])])
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- AI TRADE SIGNAL CARD ---
                st.caption("⚡ Analyzing Price Action...")
                pred, atr = train_ai(df_full)
                if pred:
                    signal = "BUY 🚀" if pred > curr else "SELL 🔻"
                    card_color = "#00d09c" if pred > curr else "#ff4b4b"
                    
                    stop_loss = curr - (1.5 * atr) if pred > curr else curr + (1.5 * atr)
                    target1 = curr + (1.0 * atr) if pred > curr else curr - (1.0 * atr)
                    target2 = curr + (2.0 * atr) if pred > curr else curr - (2.0 * atr)

                    st.markdown(f"""
                    <div class="signal-card" style="border-left: 5px solid {card_color};">
                        <div style="font-size:20px; font-weight:bold; color:{card_color}; margin-bottom:10px;">SIGNAL: {signal}</div>
                        <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:15px;">
                            <div>
                                <div style="color:#888; font-size:12px;">ENTRY PRICE</div>
                                <div style="font-size:16px; font-weight:bold;">{curr_sym}{curr:.2f}</div>
                            </div>
                            <div>
                                <div style="color:#888; font-size:12px;">STOP LOSS</div>
                                <div style="font-size:16px; font-weight:bold; color:#ff4b4b;">{curr_sym}{stop_loss:.2f}</div>
                            </div>
                            <div>
                                <div style="color:#888; font-size:12px;">TARGET 1</div>
                                <div style="font-size:16px; font-weight:bold; color:#00d09c;">{curr_sym}{target1:.2f}</div>
                            </div>
                            <div>
                                <div style="color:#888; font-size:12px;">TARGET 2</div>
                                <div style="font-size:16px; font-weight:bold; color:#00d09c;">{curr_sym}{target2:.2f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with tab2:
                try:
                    info = yf.Ticker(ticker_symbol).info
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Cap", f"{curr_sym}{info.get('marketCap',0)/10000000:.2f} Cr")
                    c2.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                    c3.metric("Sector", info.get('sector', 'N/A'))
                except: st.info("Fundamental data unavailable.")
                
                clean_sym = query.upper().replace(".NS","")
                components.html(f"""<script type="text/javascript" src="https://api.stocktwits.com/addon/widget/2/widget-loader.min.js"></script><div id="stocktwits-widget-news"></div><script type="text/javascript">STWT.Widget({{container: 'stocktwits-widget-news', symbol: '{clean_sym}', width: '100%', height: '300', limit: '15', scrollbars: 'true', streaming: 'true', title: '{clean_sym} Stream', style: {{link_color: '48515c', link_hover_color: '48515c', header_text_color: 'ffffff', border_color: '333333', divider_color: '333333', box_color: '161920', stream_color: '161920', text_color: 'ffffff', time_color: '999999'}} }});</script>""", height=320, scrolling=True)

    except Exception as e: st.error(str(e))

# --------------------------
# 9. DASHBOARD VIEW
if view == "🏠 Market Dashboard":
    st.markdown('<div class="section-title">📊 Market Indices</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    for (name, ticker), col in zip(INDICES.items(), [c1, c2, c3]):
        p, ch, pct = get_live_data(ticker)
        color = "#00d09c" if ch >= 0 else "#ff4b4b"
        symbol = "+" if ch >= 0 else ""
        with col:
            # CLICKABLE CARD BUTTON
            label = f"""
            <div style="font-size:12px; color:#aaa; font-weight:bold;">{name}</div>
            <div style="font-size:22px; font-weight:bold;">₹{p:,.2f}</div>
            <div style="color:{color}; font-weight:bold;">{symbol}{ch:.2f} ({symbol}{pct:.2f}%)</div>
            """
            if st.button(label, key=f"idx_{name}"): # The button acts as the card
                navigate_to("📈 Stock Analyzer", ticker)

    st.markdown('<div class="section-title">🔥 Most Traded (Live)</div>', unsafe_allow_html=True)
    top_vol_df = scan_most_traded()
    
    if not top_vol_df.empty:
        cols = st.columns(4)
        for i, (index, row) in enumerate(top_vol_df.iterrows()):
            sym = row['symbol'].replace(".NS","")
            pct = row['pct']
            color = "#00d09c" if pct >= 0 else "#ff4b4b"
            symbol = "+" if pct >= 0 else ""
            
            with cols[i]:
                # CLICKABLE CARD BUTTON
                label = f"""
                <div style="font-weight:bold; font-size:14px;">{sym}</div>
                <div style="font-size:18px; font-weight:bold; margin-top:2px;">₹{row['price']:.2f}</div>
                <div style="color:{color}; font-weight:bold;">{symbol}{pct:.2f}%</div>
                """
                if st.button(label, key=f"stock_{sym}"):
                    navigate_to("📈 Stock Analyzer", row['symbol'])

    st.markdown("---")
    st.markdown('<div class="section-title">📰 Latest News</div>', unsafe_allow_html=True)
    news = get_news_multi()
    for n in news:
        st.markdown(f"<a href='{n['link']}' target='_blank' style='text-decoration:none;'><div class='news-box'><div class='news-link'>{n['title']}</div><div class='news-meta'>{n['source']} • {n['time']}</div></div></a>", unsafe_allow_html=True)

# --------------------------
# 10. STOCK ANALYZER
elif view == "📈 Stock Analyzer":
    render_chart_page(st.session_state.selected_ticker, "Stock Analysis")

# --------------------------
# 11. ETF PAGE
elif view == "🏦 ETFs & Mutual Funds":
    st.markdown('<div class="section-title">🏦 Top ETFs</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (name, ticker) in enumerate(ETF_LIST.items()):
        with cols[i]:
            if st.button(name, key=f"etf_{name}"):
                st.session_state.selected_ticker = ticker
    render_chart_page(st.session_state.selected_ticker, "ETF Analysis")

# --------------------------
# 12. COMMODITIES PAGE
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
        st.metric("Spot Price", f"{spot:,.2f}")
        
        # Note: Yahoo Finance Option Data is often empty for NSE. Added safety check.
        try:
            expiries = tk.options
            if expiries:
                sel_exp = st.selectbox("Expiry", expiries)
                chain = tk.option_chain(sel_exp)
                calls, puts = chain.calls, chain.puts
                
                # PCR
                pcr = puts['openInterest'].sum() / calls['openInterest'].sum() if calls['openInterest'].sum() > 0 else 0
                st.info(f"PCR: {pcr:.2f} | Total CE OI: {calls['openInterest'].sum()} | Total PE OI: {puts['openInterest'].sum()}")

                # ATM Table
                strike_min, strike_max = spot * 0.98, spot * 1.02
                c_atm = calls[(calls['strike']>=strike_min) & (calls['strike']<=strike_max)][['strike','lastPrice','openInterest']]
                p_atm = puts[(puts['strike']>=strike_min) & (puts['strike']<=strike_max)][['strike','lastPrice','openInterest']]
                c_atm.rename(columns={'lastPrice':'CE Price','openInterest':'CE OI'}, inplace=True)
                p_atm.rename(columns={'lastPrice':'PE Price','openInterest':'PE OI'}, inplace=True)
                
                df_chain = pd.merge(c_atm, p_atm, on='strike')
                st.dataframe(df_chain.set_index('strike'), use_container_width=True)
            else:
                st.warning("Yahoo Finance returned no options data for this symbol (Common for NSE indices). Try a US symbol like 'SPY' to test the layout.")
        except Exception as e:
            st.warning(f"Could not load Option Chain: {e}")
            
    except Exception as e:
        st.error(f"Error: {e}")

# --------------------------
# 14. AI PICKS
elif view == "⭐ Top 5 AI Picks":
    st.markdown('<div class="section-title">⭐ AI Market Scanners</div>', unsafe_allow_html=True)
    st.caption("Scanning market leaders...")
    
    results = []
    progress = st.progress(0)
    for i, ticker in enumerate(SCANNER_POOL[:6]):
        progress.progress((i+1)/6)
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
            <div class="news-box" style="border-left: 5px solid {color}">
                <h3>{sym.replace('.NS','')}</h3>
                <p>Price: ₹{curr:.2f} | Signal: <b>{action}</b> (Potential: {diff:+.2f})</p>
            </div>
            """, unsafe_allow_html=True)
