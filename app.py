import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
import feedparser
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# --------------------------
# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Pro Market Dashboard", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="collapsed"
)

# --------------------------
# 2. SESSION STATE
if 'page' not in st.session_state: st.session_state.page = "🏠 Market Dashboard"
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = "RELIANCE.NS"

def navigate_to(page, ticker=None):
    st.session_state.page = page
    if ticker: st.session_state.selected_ticker = ticker
    st.rerun()

# --------------------------
# 3. CSS STYLING
st.markdown("""
<style>
    .stApp { background-color: #0f1115; font-family: 'Inter', sans-serif; }
    
    /* CARDS */
    .fun-card {
        background: rgba(30, 34, 45, 0.6); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; padding: 15px; cursor: pointer; transition: 0.3s;
    }
    .fun-card:hover { transform: translateY(-3px); border-color: #00d09c; }
    
    /* NEWS */
    .news-box { background: #161920; padding: 12px; border-radius: 10px; margin-bottom: 8px; border-left: 3px solid #4c8bf5; }
    
    /* BUTTONS */
    div.stButton > button { width: 100%; background: transparent; border: none; color: white; text-align: left; padding: 0; }
    div.stButton > button:hover { color: #00d09c; background: transparent; }
    
    /* TABLES */
    .stDataFrame { border: 1px solid #333; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. ROBUST DATA FETCHING (THE FIX)
def robust_yf_download(ticker, period):
    # This function handles the "Blank Graph" issue
    try:
        df = yf.download(ticker, period=period, progress=False)
        
        # FIX: Flatten MultiIndex columns (e.g. ('Close', 'RELIANCE.NS') -> 'Close')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure we have data
        if df.empty: return None
        return df
    except: return None

@st.cache_data(ttl=60)
def get_nse_chain(symbol):
    # Direct NSE Connection for F/O Data
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        
        clean_sym = symbol.upper().replace(".NS","").replace("^","")
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={clean_sym}" if clean_sym in ["NIFTY","BANKNIFTY"] else f"https://www.nseindia.com/api/option-chain-equities?symbol={clean_sym}"
        
        return session.get(url, headers=headers, timeout=5).json()
    except: return None

def get_currency_symbol(ticker):
    if ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^"): return "₹"
    return "$"

def add_indicators(df):
    if df.empty: return df
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean() # Volatility
    return df.fillna(0)

# AI Signal Generator
def generate_ai_signal(df):
    if len(df) < 50: return None
    
    # Simple Strategy: Price vs SMA + Trend
    curr = df['Close'].iloc[-1]
    sma = df['SMA_50'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    signal = "BUY" if curr > sma else "SELL"
    
    # Calculate Levels
    if signal == "BUY":
        stop_loss = curr - (1.5 * atr)
        target = curr + (2.0 * atr)
    else:
        stop_loss = curr + (1.5 * atr)
        target = curr - (2.0 * atr)
        
    return signal, stop_loss, target

# --------------------------
# 5. DATA POOLS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS", "ZOMATO.NS"]

ETF_LIST = {
    "Nifty BeES": "NIFTYBEES.NS", "Bank BeES": "BANKBEES.NS", 
    "Silver BeES": "SILVERBEES.NS", "Gold BeES": "GOLDBEES.NS", 
    "IT BeES": "ITBEES.NS", "Pharma BeES": "PHARMABEES.NS"
}

COMMODITY_LIST = {
    "Gold ($)": "GC=F", "Silver ($)": "SI=F", 
    "Crude Oil ($)": "CL=F", "Natural Gas ($)": "NG=F", "Copper ($)": "HG=F"
}

# --------------------------
# 6. SIDEBAR
st.sidebar.title("🚀 Menu")
nav = st.sidebar.radio("Go to", ["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"], index=["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"].index(st.session_state.page))
if nav != st.session_state.page: navigate_to(nav)

if st.session_state.page != "⭐ Top 5 AI Picks":
    st_autorefresh(interval=30000, key="refresh")

# --------------------------
# 7. DASHBOARD VIEW
if st.session_state.page == "🏠 Market Dashboard":
    st.markdown("### 📊 Market Overview")
    c1, c2, c3 = st.columns(3)
    for (name, sym), col in zip(INDICES.items(), [c1, c2, c3]):
        df = robust_yf_download(sym, "5d")
        if df is not None:
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            chg = curr - prev
            pct = (chg/prev)*100
            clr = "#00d09c" if chg >= 0 else "#ff4b4b"
            with col:
                st.markdown(f"""<div class="fun-card" style="border-top:3px solid {clr}"><div style="color:#aaa; font-size:12px;">{name}</div><div style="font-size:22px; font-weight:bold;">₹{curr:,.2f}</div><div style="color:{clr}; font-weight:bold;">{chg:+.2f} ({pct:+.2f}%)</div></div>""", unsafe_allow_html=True)
                if st.button(f"Analyze {name}", key=name): navigate_to("📈 Stock Analyzer", sym)

    st.markdown("### 📰 Latest News")
    st.info("Fetching live news...")
    # (Simplified news fetcher for reliability)
    try:
        d = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
        for e in d.entries[:4]:
            st.markdown(f"<div class='news-box'><a href='{e.link}' target='_blank' style='color:white;text-decoration:none'>{e.title}</a></div>", unsafe_allow_html=True)
    except: st.error("News feed unavailable")

# --------------------------
# 8. F/O DASHBOARD (NSE NATIVE)
elif st.session_state.page == "📊 F/O Dashboard":
    st.markdown("### 📊 NSE Option Chain")
    c1, c2 = st.columns([1, 2])
    fo_sym = c1.text_input("Symbol", "NIFTY")
    
    data = get_nse_chain(fo_sym)
    
    if data:
        try:
            recs = data['records']['data']
            exps = data['records']['expiryDates']
            sel_exp = st.selectbox("Expiry", exps)
            chain = [x for x in recs if x['expiryDate'] == sel_exp]
            spot = data['records']['underlyingValue']
            
            st.metric(f"{fo_sym} Spot", f"₹{spot:,.2f}")
            
            ce_oi = sum([x['CE']['openInterest'] for x in chain if 'CE' in x])
            pe_oi = sum([x['PE']['openInterest'] for x in chain if 'PE' in x])
            pcr = pe_oi / ce_oi if ce_oi > 0 else 0
            
            k1, k2, k3 = st.columns(3)
            k1.metric("PCR", f"{pcr:.2f}")
            k2.metric("Sentiment", "BULLISH 🐂" if pcr>1 else "BEARISH 🐻")
            k3.metric("Total OI", f"{ce_oi+pe_oi:,.0f}")
            
            # Table
            rows = []
            for x in chain:
                if x['strikePrice'] > spot*0.98 and x['strikePrice'] < spot*1.02:
                    row = {'Strike': x['strikePrice']}
                    if 'CE' in x: row.update({'CE Price': x['CE']['lastPrice'], 'CE OI': x['CE']['openInterest']})
                    if 'PE' in x: row.update({'PE Price': x['PE']['lastPrice'], 'PE OI': x['PE']['openInterest']})
                    rows.append(row)
            
            st.dataframe(pd.DataFrame(rows).set_index('Strike'), use_container_width=True)
        except: st.error("Could not parse NSE data. Market might be closed or API changed.")
    else:
        st.error("Connection to NSE failed. This usually happens on Cloud servers due to IP blocking. Try running locally.")

# --------------------------
# 9. STOCK ANALYZER (FIXED CHARTS & AI)
elif st.session_state.page == "📈 Stock Analyzer":
    st.markdown("### 📈 Analyzer")
    if st.button("← Back"): navigate_to("🏠 Market Dashboard")
    
    c1, c2 = st.columns([1,3])
    ex = c1.selectbox("Market", ["NSE","BSE","Global"])
    clean_default = st.session_state.selected_ticker.replace(".NS","").replace(".BO","")
    sym = c2.text_input("Ticker", clean_default)
    
    if ex == "NSE": full = f"{sym}.NS"
    elif ex == "BSE": full = f"{sym}.BO"
    else: full = sym 
    
    curr_sym = get_currency_symbol(full)
    
    # 1. TIMEFRAME SELECTOR
    t_tabs = st.tabs(["1M", "3M", "6M", "1Y", "MAX"])
    time_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "MAX": "5y"}
    
    # We load based on the first tab initially, but in reality, let's just load 1Y data and slice it for speed
    df = robust_yf_download(full, "2y")
    
    if df is not None:
        df = add_indicators(df)
        curr = df['Close'].iloc[-1]
        
        # Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. AI TRADE SIGNALS
        res = generate_ai_signal(df)
        if res:
            sig, sl, tgt = res
            clr = "#00d09c" if sig=="BUY" else "#ff4b4b"
            
            st.markdown(f"""
            <div style="background:#1e2330; border:1px solid {clr}; border-radius:12px; padding:20px; margin-top:20px;">
                <h2 style="color:{clr}; margin:0;">{sig} SIGNAL 🚀</h2>
                <hr style="border-color:#333;">
                <div style="display:flex; justify-content:space-between;">
                    <div><span style="color:#888;">Entry</span><br><h3>{curr_sym}{curr:.2f}</h3></div>
                    <div><span style="color:#ff4b4b;">Stop Loss</span><br><h3>{curr_sym}{sl:.2f}</h3></div>
                    <div><span style="color:#00d09c;">Target</span><br><h3>{curr_sym}{tgt:.2f}</h3></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 3. STOCKTWITS (Fixed)
        clean_twit = sym.replace(".NS","")
        components.html(f"""<script type="text/javascript" src="https://api.stocktwits.com/addon/widget/2/widget-loader.min.js"></script><div id="stocktwits-widget-news"></div><script type="text/javascript">STWT.Widget({{container: 'stocktwits-widget-news', symbol: '{clean_twit}', width: '100%', height: '300', limit: '15', scrollbars: 'true', streaming: 'true', title: '{clean_twit} Stream', style: {{link_color: '48515c', link_hover_color: '48515c', header_text_color: 'ffffff', border_color: '333333', divider_color: '333333', box_color: '161920', stream_color: '161920', text_color: 'ffffff', time_color: '999999'}} }});</script>""", height=320, scrolling=True)
        
        # 4. FUNDAMENTALS
        try:
            info = yf.Ticker(full).info
            f1, f2, f3 = st.columns(3)
            f1.metric("Market Cap", f"{curr_sym}{info.get('marketCap',0)/10000000:.2f} Cr")
            f2.metric("P/E", f"{info.get('trailingPE',0):.2f}")
            f3.metric("Sector", info.get('sector','N/A'))
        except: st.warning("Fundamentals unavailable via API.")

    else: st.error(f"No data found for {full}. Try checking the symbol.")

# --------------------------
# 10. ETF & COMMODITIES (FIXED)
elif st.session_state.page == "🏦 ETFs & Commodities":
    st.markdown("### 🏦 ETFs & Global Commodities")
    type_ = st.radio("Asset Class", ["ETFs (India)", "Commodities (Global)"], horizontal=True)
    
    active_list = ETF_LIST if type_ == "ETFs (India)" else COMMODITY_LIST
    
    cols = st.columns(4)
    for i, (name, ticker) in enumerate(active_list.items()):
        with cols[i % 4]:
            if st.button(name, key=name):
                navigate_to("📈 Stock Analyzer", ticker)

# --------------------------
# 11. AI PICKS
elif st.session_state.page == "⭐ Top 5 AI Picks":
    st.markdown("### ⭐ AI Scanners")
    if st.button("Start Scan"):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(SCANNER_POOL[:5]):
            bar.progress((i+1)/5)
            df = robust_yf_download(t, "1y")
            if df is not None:
                df = add_indicators(df)
                res = generate_ai_signal(df)
                if res: results.append((t, res[0], df['Close'].iloc[-1]))
        bar.empty()
        for t, sig, curr in results:
            clr = "#00d09c" if sig=="BUY" else "#ff4b4b"
            st.markdown(f"<div class='fun-card' style='border-left:5px solid {clr}'><b>{t}</b>: {sig} @ ₹{curr:.2f}</div>", unsafe_allow_html=True)
