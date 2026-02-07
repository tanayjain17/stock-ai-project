import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import feedparser
from textblob import TextBlob
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import time

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
    
    /* BUTTON OVERRIDES */
    div.stButton > button { width: 100%; background: transparent; border: none; color: white; text-align: left; padding: 0; }
    div.stButton > button:hover { color: #00d09c; background: transparent; }
    
    /* TABLES */
    .stDataFrame { border: 1px solid #333; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. DATA POOLS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS", "ZOMATO.NS"]

# NEW: EXPANDED LISTS
ETF_LIST = {
    "Nifty BeES": "NIFTYBEES.NS", 
    "Bank BeES": "BANKBEES.NS", 
    "Silver BeES": "SILVERBEES.NS", 
    "Gold BeES": "GOLDBEES.NS", 
    "IT BeES": "ITBEES.NS", 
    "Pharma BeES": "PHARMABEES.NS"
}

COMMODITY_LIST = {
    "Gold (Global)": "GC=F", 
    "Silver (Global)": "SI=F", 
    "Crude Oil": "CL=F", 
    "Natural Gas": "NG=F", 
    "Copper": "HG=F"
}

# --------------------------
# 5. UTILITIES (Currency Fix)
def get_currency_symbol(ticker):
    # If it ends with .NS, .BO or starts with ^, it's Indian Rupee
    if ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^"):
        return "₹"
    # Otherwise (Global Commodities/US Stocks), it's Dollar
    return "$"

def get_live_price(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.fast_info.last_price, t.fast_info.previous_close
    except: return 0,0

@st.cache_data(ttl=300)
def get_news():
    feeds = ["https://www.moneycontrol.com/rss/marketreports.xml", "https://www.livemint.com/rss/markets"]
    items = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:3]:
                items.append({'title':e.title, 'link':e.link, 'src': url.split('.')[1]})
        except: continue
    return items

@st.cache_data(ttl=60)
def get_nse_chain(symbol="NIFTY"):
    # Official NSE Scraper (Same as V6.0)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36','Accept-Encoding': 'gzip, deflate, br','Accept-Language': 'en-US,en;q=0.9'}
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        clean_sym = symbol.upper().replace(".NS","").replace("^","")
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={clean_sym}" if clean_sym in ["NIFTY","BANKNIFTY"] else f"https://www.nseindia.com/api/option-chain-equities?symbol={clean_sym}"
        return session.get(url, headers=headers, timeout=10).json()
    except: return None

def run_ai_scan(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if len(df) < 50: return None
        df['SMA'] = df['Close'].rolling(50).mean()
        curr = df['Close'].iloc[-1]
        sma = df['SMA'].iloc[-1]
        signal = "BUY" if curr > sma else "SELL"
        target = curr * 1.05 if signal == "BUY" else curr * 0.95
        return signal, target
    except: return None

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
        curr, prev = get_live_price(sym)
        chg = curr - prev
        pct = (chg/prev)*100 if prev else 0
        clr = "#00d09c" if chg >= 0 else "#ff4b4b"
        with col:
            st.markdown(f"""<div class="fun-card" style="border-top:3px solid {clr}"><div style="color:#aaa; font-size:12px;">{name}</div><div style="font-size:22px; font-weight:bold;">₹{curr:,.2f}</div><div style="color:{clr}; font-weight:bold;">{chg:+.2f} ({pct:+.2f}%)</div></div>""", unsafe_allow_html=True)
            if st.button(f"Analyze {name}", key=name): navigate_to("📈 Stock Analyzer", sym)

    st.markdown("### 📰 Latest News")
    news = get_news()
    for n in news:
        st.markdown(f"<div class='news-box'><a href='{n['link']}' target='_blank' style='color:white;text-decoration:none'>{n['title']}</a> <span style='color:#888;font-size:10px'>- {n['src']}</span></div>", unsafe_allow_html=True)

# --------------------------
# 8. F/O DASHBOARD
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
                if x['strikePrice'] > spot*0.97 and x['strikePrice'] < spot*1.03:
                    row = {'Strike': x['strikePrice']}
                    if 'CE' in x: row.update({'CE LTP': x['CE']['lastPrice'], 'CE OI': x['CE']['openInterest']})
                    if 'PE' in x: row.update({'PE LTP': x['PE']['lastPrice'], 'PE OI': x['PE']['openInterest']})
                    rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index('Strike'), use_container_width=True)
        except: st.error("Error parsing NSE data.")
    else: st.error("NSE Connection Failed (Try Localhost).")

# --------------------------
# 9. STOCK ANALYZER (Universal)
elif st.session_state.page == "📈 Stock Analyzer":
    st.markdown("### 📈 Analyzer")
    if st.button("← Back"): navigate_to("🏠 Market Dashboard")
    
    c1, c2 = st.columns([1,3])
    ex = c1.selectbox("Market", ["NSE","BSE","Global"])
    clean_default = st.session_state.selected_ticker.replace(".NS","").replace(".BO","")
    sym = c2.text_input("Ticker", clean_default)
    
    if ex == "NSE": full = f"{sym}.NS"
    elif ex == "BSE": full = f"{sym}.BO"
    else: full = sym # Global
    
    # Currency Check
    curr_sym = get_currency_symbol(full)
    
    df = yf.download(full, period="1y", progress=False)
    if not df.empty:
        curr = df['Close'].iloc[-1]
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        res = run_ai_scan(full)
        if res:
            sig, tgt = res
            clr = "#00d09c" if sig=="BUY" else "#ff4b4b"
            st.markdown(f"""<div style="background:{clr}20; border-left:5px solid {clr}; padding:15px; border-radius:10px;">
                <h3 style="margin:0; color:{clr}">{sig} SIGNAL</h3>
                <p style="margin:0">Current: {curr_sym}{curr:.2f} | Target: <b>{curr_sym}{tgt:.2f}</b></p></div>""", unsafe_allow_html=True)
    else: st.error("Ticker not found.")

# --------------------------
# 10. ETF & COMMODITIES (With Correct Currency)
elif st.session_state.page == "🏦 ETFs & Commodities":
    st.markdown("### 🏦 ETFs & Global Commodities")
    type_ = st.radio("Asset Class", ["ETFs (India)", "Commodities (Global)"], horizontal=True)
    
    # Select List based on radio
    active_list = ETF_LIST if type_ == "ETFs (India)" else COMMODITY_LIST
    
    # Grid Buttons
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
            res = run_ai_scan(t)
            if res: results.append((t, res[0], res[1]))
        bar.empty()
        for t, sig, tgt in results:
            clr = "#00d09c" if sig=="BUY" else "#ff4b4b"
            st.markdown(f"<div class='fun-card' style='border-left:5px solid {clr}'><b>{t}</b>: {sig} -> Tgt {tgt:.1f}</div>", unsafe_allow_html=True)
