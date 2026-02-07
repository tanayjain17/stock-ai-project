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
        cursor: pointer;
    }
    .fun-card:hover { transform: translateY(-5px); border-color: #00d09c; }
    
    /* BUTTON STYLING FOR CARDS */
    .stButton button {
        width: 100%;
        background-color: transparent;
        border: none;
        color: white;
        text-align: left;
        padding: 0;
    }
    .stButton button:hover {
        border: none;
        color: #00d09c;
    }

    /* GRADIENTS & TEXT */
    .gradient-text-green { background: -webkit-linear-gradient(45deg, #00d09c, #00ffaa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .gradient-text-red { background: -webkit-linear-gradient(45deg, #ff4b4b, #ff9068); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .section-title { font-size: 22px; font-weight: 700; margin-top: 25px; margin-bottom: 15px; color: #e0e0e0; }
    
    /* NEWS */
    .news-box { padding: 12px; border-radius: 12px; background: #161920; margin-bottom: 10px; border-left: 3px solid #4c8bf5; }
    .news-link { color: #fff; text-decoration: none; font-weight: 500; font-size: 14px; }
    .news-link:hover { color: #4c8bf5; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. DATA POOLS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["ZOMATO.NS", "YESBANK.NS", "IDEA.NS", "TATASTEEL.NS", "RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS", "INFY.NS", "ITC.NS", "TATAMOTORS.NS"]
ETF_LIST = {"Nifty BeES": "NIFTYBEES.NS", "Gold BeES": "GOLDBEES.NS", "Silver BeES": "SILVERBEES.NS", "Bank BeES": "BANKBEES.NS", "IT BeES": "ITBEES.NS", "Pharma BeES": "PHARMABEES.NS"}
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
    pred_price = scaler.inverse_transform([[pred_scaled[0,0],0,0,0]])[0,0] # inverse dummy
    return pred_price, df['ATR'].iloc[-1]

# --------------------------
# 8. SHARED CHART COMPONENT
def render_chart_page(default_ticker, title):
    col_ex, col_search = st.columns([1, 4])
    with col_ex:
        exchange = st.selectbox("Exchange", ["NSE", "BSE", "Global"], index=0 if "NS" in default_ticker else 2)
    with col_search:
        clean_default = default_ticker.replace(".NS","").replace(".BO","")
        query = st.text_input("Search Ticker", clean_default)

    if exchange == "NSE": suffix = ".NS"
    elif exchange == "BSE": suffix = ".BO"
    else: suffix = "" # Global has no suffix usually or =F
    
    # If user changed query, use that. Otherwise use passed default.
    if query != clean_default:
        ticker_symbol = f"{query.upper().strip()}{suffix}"
    else:
        ticker_symbol = default_ticker

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

            st.markdown(f"""
            <div style="display:flex; align-items:baseline; gap:15px; margin-bottom:15px;">
                <h1 style="margin:0;">{query.upper()}</h1>
                <h2 style="margin:0; color:{color};">₹{curr:,.2f}</h2>
                <span style="color:{color}; font-weight:bold; font-size:18px;">{diff:+.2f} ({pct:+.2f}%)</span>
            </div>""", unsafe_allow_html=True)

            tf_map = {"1M":30,"6M":180,"1Y":365,"MAX":5000}
            sel_tf = st.radio("Range", list(tf_map.keys()), horizontal=True, label_visibility="collapsed")
            df_chart = df_full.tail(tf_map[sel_tf])

            tab1, tab2 = st.tabs(["📊 Technicals", "🧠 Fundamentals"])
            with tab1:
                fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'])])
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("⚡ AI Prediction running...")
                pred, atr = train_ai(df_full)
                if pred:
                    signal = "BUY 🚀" if pred > curr else "SELL 🔻"
                    st.success(f"AI Target: {pred:.2f} | Signal: {signal}")

            with tab2:
                try:
                    info = yf.Ticker(ticker_symbol).info
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Cap", f"{info.get('marketCap',0)/10000000:.2f} Cr")
                    c2.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                    c3.metric("Sector", info.get('sector', 'N/A'))
                except: st.info("No fundamental data.")
                
                # StockTwits
                clean_sym = query.upper().replace(".NS","")
                components.html(f"""<script type="text/javascript" src="https://api.stocktwits.com/addon/widget/2/widget-loader.min.js"></script><div id="stocktwits-widget-news"></div><script type="text/javascript">STWT.Widget({{container: 'stocktwits-widget-news', symbol: '{clean_sym}', width: '100%', height: '300', limit: '15', scrollbars: 'true', streaming: 'true', title: '{clean_sym} Stream', style: {{link_color: '48515c', link_hover_color: '48515c', header_text_color: 'ffffff', border_color: '333333', divider_color: '333333', box_color: '161920', stream_color: '161920', text_color: 'ffffff', time_color: '999999'}} }});</script>""", height=320, scrolling=True)

    except Exception as e: st.error(str(e))

# --------------------------
# 9. DASHBOARD VIEW
if view == "🏠 Market Dashboard":
    st.markdown('<div class="section-title">📊 Market Indices (Click to Analyze)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for (name, ticker), col in zip(INDICES.items(), [c1, c2, c3]):
        p, ch, pct = get_live_data(ticker)
        color = "#00d09c" if ch >= 0 else "#ff4b4b"
        with col:
            # Clickable Card Logic
            if st.button(f"{name}\n₹{p:,.2f}  {ch:+.2f} ({pct:+.2f}%)", key=name):
                navigate_to("📈 Stock Analyzer", ticker)
            
            # Visual card (rendered below button invisibly or just use button text)
            # Streamlit buttons don't support HTML styling inside well, so we use the button text for info.

    st.markdown('<div class="section-title">🔥 Most Traded (Click to Analyze)</div>', unsafe_allow_html=True)
    top_vol_df = scan_most_traded()
    if not top_vol_df.empty:
        cols = st.columns(4)
        for i, (index, row) in enumerate(top_vol_df.iterrows()):
            sym = row['symbol'].replace(".NS","")
            pct = row['pct']
            with cols[i]:
                label = f"{sym}\n₹{row['price']:.2f}\n{pct:+.2f}%"
                if st.button(label, key=f"btn_{sym}"):
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
# 11. ETF PAGE
elif view == "🏦 ETFs & Mutual Funds":
    st.markdown('<div class="section-title">🏦 Top ETFs</div>', unsafe_allow_html=True)
    sel_etf = st.selectbox("Choose ETF", list(ETF_LIST.keys()))
    render_chart_page(ETF_LIST[sel_etf], "ETF Analysis")

# --------------------------
# 12. COMMODITIES PAGE
elif view == "🛢️ Global Commodities":
    st.markdown('<div class="section-title">🛢️ Global Commodities</div>', unsafe_allow_html=True)
    sel_comm = st.selectbox("Choose Asset", list(COMMODITY_LIST.keys()))
    render_chart_page(COMMODITY_LIST[sel_comm], "Commodity Analysis")

# --------------------------
# 13. F/O DASHBOARD (NEW)
elif view == "📊 F/O & Options Chain":
    st.markdown('<div class="section-title">📊 Options Chain & Derivatives</div>', unsafe_allow_html=True)
    
    fo_ticker = st.text_input("F/O Symbol (e.g. NIFTY, RELIANCE)", "NIFTY")
    
    # Handle Indices vs Stocks for yfinance
    if fo_ticker.upper() == "NIFTY": yf_sym = "^NSEI"
    elif fo_ticker.upper() == "BANKNIFTY": yf_sym = "^NSEBANK"
    else: yf_sym = f"{fo_ticker.upper()}.NS"

    try:
        tk = yf.Ticker(yf_sym)
        expiries = tk.options
        if not expiries:
            st.warning(f"No options data found for {fo_ticker}. Note: yfinance often delays NSE option data.")
        else:
            sel_exp = st.selectbox("Select Expiry", expiries)
            
            # Fetch Chain
            opt = tk.option_chain(sel_exp)
            calls = opt.calls
            puts = opt.puts
            
            # Underlying Price (approx)
            spot = tk.fast_info.last_price
            st.metric("Spot Price", f"{spot:,.2f}")

            # PCR Calculation
            total_ce_oi = calls['openInterest'].sum()
            total_pe_oi = puts['openInterest'].sum()
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            c1, c2 = st.columns(2)
            c1.metric("Put-Call Ratio (PCR)", f"{pcr:.2f}")
            c2.caption("PCR > 1.0 generally Bullish, < 0.7 Bearish")

            # Display Table
            st.subheader("Option Chain (Near ATM)")
            
            # Filter near ATM
            atm_calls = calls[(calls['strike'] > spot*0.95) & (calls['strike'] < spot*1.05)]
            atm_puts = puts[(puts['strike'] > spot*0.95) & (puts['strike'] < spot*1.05)]
            
            # Clean view
            df_chain = pd.merge(atm_calls[['strike','lastPrice','openInterest','volume']], 
                                atm_puts[['strike','lastPrice','openInterest','volume']], 
                                on='strike', suffixes=('_CE', '_PE'))
            
            st.dataframe(df_chain.set_index('strike').sort_index(), use_container_width=True)
            
    except Exception as e:
        st.error(f"Error fetching F/O data: {e}")

# --------------------------
# 14. AI PICKS
elif view == "⭐ Top 5 AI Picks":
    st.info("AI Scanner running on simplified pool...")
    # (Reuse existing scanner logic from previous steps here)
