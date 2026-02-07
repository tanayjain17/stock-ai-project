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
    .fun-card {
        background: rgba(30, 34, 45, 0.6); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; padding: 15px; cursor: pointer; transition: 0.3s;
    }
    .fun-card:hover { transform: translateY(-3px); border-color: #00d09c; }
    .news-box { background: #161920; padding: 12px; border-radius: 10px; margin-bottom: 8px; border-left: 3px solid #4c8bf5; }
    div.stButton > button { width: 100%; background: transparent; border: none; color: white; text-align: left; padding: 0; }
    div.stButton > button:hover { color: #00d09c; background: transparent; }
    .stDataFrame { border: 1px solid #333; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. ROBUST NSE CONNECTION (The "Magic" Scraper)
@st.cache_data(ttl=60) # Cache for 60s to avoid being blocked
def get_nse_chain(symbol="NIFTY"):
    # Headers to mimic a real browser visiting NSE
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    session = requests.Session()
    try:
        # 1. Visit Homepage first to get "Cookies" (Crucial step)
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        
        # 2. Determine Correct API URL
        clean_sym = symbol.upper().replace(".NS","").replace("^","")
        if clean_sym in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={clean_sym}"
        else:
            # For stocks, NSE uses a different endpoint format
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={clean_sym}"
            
        # 3. Fetch Data with Cookies
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# --------------------------
# 5. DATA UTILITIES
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS", "ZOMATO.NS"]

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
# 6. SIDEBAR & NAVIGATION
st.sidebar.title("🚀 Menu")
nav = st.sidebar.radio("Go to", ["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"], index=["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"].index(st.session_state.page))
if nav != st.session_state.page: navigate_to(nav)

if st.session_state.page != "⭐ Top 5 AI Picks":
    st_autorefresh(interval=30000, key="refresh")

# --------------------------
# 7. VIEW: DASHBOARD
if st.session_state.page == "🏠 Market Dashboard":
    st.markdown("### 📊 Market Overview")
    c1, c2, c3 = st.columns(3)
    for (name, sym), col in zip(INDICES.items(), [c1, c2, c3]):
        curr, prev = get_live_price(sym)
        chg = curr - prev
        pct = (chg/prev)*100
        clr = "#00d09c" if chg >= 0 else "#ff4b4b"
        with col:
            st.markdown(f"""
            <div class="fun-card" style="border-top:3px solid {clr}">
                <div style="color:#aaa; font-size:12px;">{name}</div>
                <div style="font-size:22px; font-weight:bold;">₹{curr:,.2f}</div>
                <div style="color:{clr}; font-weight:bold;">{chg:+.2f} ({pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Analyze {name}", key=name): navigate_to("📈 Stock Analyzer", sym)

    st.markdown("### 📰 Latest News")
    news = get_news()
    for n in news:
        st.markdown(f"<div class='news-box'><a href='{n['link']}' target='_blank' style='color:white;text-decoration:none'>{n['title']}</a> <span style='color:#888;font-size:10px'>- {n['src']}</span></div>", unsafe_allow_html=True)

# --------------------------
# 8. VIEW: F/O DASHBOARD (THE NEW UPGRADE)
elif st.session_state.page == "📊 F/O Dashboard":
    st.markdown("### 📊 NSE Option Chain (Live)")
    
    # Input
    col1, col2 = st.columns([1, 2])
    with col1:
        fo_sym = st.text_input("Symbol", "NIFTY")
    with col2:
        st.info("Fetching directly from NSE India API...")

    # Fetch
    data = get_nse_chain(fo_sym)
    
    if data:
        try:
            records = data['records']['data']
            expiry_list = data['records']['expiryDates']
            sel_exp = st.selectbox("Select Expiry", expiry_list)
            
            # Filter by Expiry
            chain = [x for x in records if x['expiryDate'] == sel_exp]
            
            # Underlying Price
            spot = data['records']['underlyingValue']
            st.metric(f"{fo_sym.upper()} Spot", f"₹{spot:,.2f}")
            
            # Build DataFrame
            ce_data = []
            pe_data = []
            total_ce_oi = 0
            total_pe_oi = 0
            
            for item in chain:
                strike = item['strikePrice']
                # CE Data
                if 'CE' in item:
                    ce = item['CE']
                    ce_data.append({'Strike': strike, 'CE LTP': ce['lastPrice'], 'CE OI': ce['openInterest'], 'CE Vol': ce['totalTradedVolume']})
                    total_ce_oi += ce['openInterest']
                # PE Data
                if 'PE' in item:
                    pe = item['PE']
                    pe_data.append({'Strike': strike, 'PE LTP': pe['lastPrice'], 'PE OI': pe['openInterest'], 'PE Vol': pe['totalTradedVolume']})
                    total_pe_oi += pe['openInterest']

            # Analysis
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            sentiment = "BULLISH 🐂" if pcr > 1 else "BEARISH 🐻" if pcr < 0.7 else "NEUTRAL ⚖️"
            
            c1, c2, c3 = st.columns(3)
            c1.metric("PCR (Put Call Ratio)", f"{pcr:.2f}")
            c2.metric("Market Sentiment", sentiment)
            c3.metric("Total OI", f"{total_ce_oi+total_pe_oi:,.0f}")

            # Merge & Display Table (Near ATM)
            df_ce = pd.DataFrame(ce_data)
            df_pe = pd.DataFrame(pe_data)
            df_chain = pd.merge(df_ce, df_pe, on='Strike')
            
            # Filter range (Spot +/- 3%)
            df_chain = df_chain[(df_chain['Strike'] > spot*0.97) & (df_chain['Strike'] < spot*1.03)]
            
            st.markdown("#### Option Chain (Near Spot)")
            st.dataframe(df_chain.set_index('Strike').style.highlight_max(axis=0, subset=['CE OI', 'PE OI'], color='#1e2330'), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error parsing NSE data: {e}")
    else:
        st.error("Failed to connect to NSE. If on Cloud, try Local Machine.")
        st.caption("Why? NSE blocks cloud server IPs. This feature works best on your laptop (Localhost).")

# --------------------------
# 9. VIEW: STOCK ANALYZER
elif st.session_state.page == "📈 Stock Analyzer":
    st.markdown("### 📈 Technical Analyzer")
    if st.button("← Back"): navigate_to("🏠 Market Dashboard")
    
    col1, col2 = st.columns([1,3])
    ex = col1.selectbox("Exch", ["NSE","BSE"])
    sym = col2.text_input("Ticker", st.session_state.selected_ticker.replace(".NS",""))
    full_sym = f"{sym}.NS" if ex == "NSE" else f"{sym}.BO"
    
    df = yf.download(full_sym, period="1y", progress=False)
    if not df.empty:
        # Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Card
        res = run_ai_scan(full_sym)
        if res:
            sig, tgt = res
            clr = "#00d09c" if sig=="BUY" else "#ff4b4b"
            st.markdown(f"""
            <div style="background:{clr}20; border-left:5px solid {clr}; padding:15px; border-radius:10px; margin-top:10px;">
                <h3 style="margin:0; color:{clr}">{sig} SIGNAL</h3>
                <p style="margin:0">Target: <b>₹{tgt:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Ticker not found.")

# --------------------------
# 10. VIEW: ETF/COMMODITIES
elif st.session_state.page == "🏦 ETFs & Commodities":
    st.markdown("### 🏦 ETFs & Global Commodities")
    type_ = st.radio("Type", ["ETFs", "Commodities"], horizontal=True)
    
    list_ = {"Nifty BeES": "NIFTYBEES.NS", "Gold BeES": "GOLDBEES.NS"} if type_=="ETFs" else {"Gold ($)": "GC=F", "Oil ($)": "CL=F"}
    
    sel = st.selectbox("Select Asset", list(list_.keys()))
    t = list_[sel]
    
    df = yf.download(t, period="6mo", progress=False)
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 11. VIEW: AI PICKS
elif st.session_state.page == "⭐ Top 5 AI Picks":
    st.markdown("### ⭐ AI Market Scanners")
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
