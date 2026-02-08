import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import requests
import feedparser
from bs4 import BeautifulSoup
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
    
    /* METRIC BOX */
    .metric-container {
        background-color: #1e2330;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #333;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 11px; color: #aaa; text-transform: uppercase; }
    .metric-value { font-size: 16px; font-weight: bold; color: #fff; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. DATA ENGINES (Robust & Smart)

# A. Chart Data Fix (Flattens Columns)
def robust_yf_download(ticker, period):
    try:
        # Smart Interval Selection
        interval = "1m" if period in ["1d","5d"] else "1d"
        if period == "1mo": interval = "30m"
        
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Flatten MultiIndex (The Blank Chart Fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df if not df.empty else None
    except: return None

# B. NSE Option Chain (Direct)
@st.cache_data(ttl=60)
def get_nse_chain(symbol):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        clean_sym = symbol.upper().replace(".NS","").replace("^","")
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={clean_sym}" if clean_sym in ["NIFTY","BANKNIFTY"] else f"https://www.nseindia.com/api/option-chain-equities?symbol={clean_sym}"
        return session.get(url, headers=headers, timeout=5).json()
    except: return None

# C. Fundamental Data (Screener.in Fallback) - NEW!
@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    # 1. Try Yahoo Finance First (Fastest)
    try:
        info = yf.Ticker(ticker).info
        if info.get('trailingPE') is not None:
            return {
                "Market Cap": info.get("marketCap", 0),
                "P/E Ratio": info.get("trailingPE", 0),
                "P/B Ratio": info.get("priceToBook", 0),
                "ROE": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
                "Book Value": info.get("bookValue", 0),
                "Div Yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "Industry": info.get("industry", "N/A"),
                "Source": "Yahoo"
            }
    except: pass

    # 2. Fallback to Screener.in (Reliable for India)
    # Only works for Indian stocks (.NS or .BO)
    if ".NS" in ticker or ".BO" in ticker:
        try:
            clean_sym = ticker.replace(".NS","").replace(".BO","")
            url = f"https://www.screener.in/company/{clean_sym}/"
            r = requests.get(url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                ratios = soup.find('ul', {'id': 'top-ratios'})
                data = {}
                
                # Helper to find text in li
                def get_val(name):
                    for li in ratios.find_all('li'):
                        if name in li.find('span', {'class':'name'}).text:
                            val = li.find('span', {'class':'number'}).text.replace(',','')
                            return float(val) if val else 0
                    return 0

                data["Market Cap"] = get_val("Market Cap") * 10000000 # Convert Cr to Raw
                data["P/E Ratio"] = get_val("Stock P/E")
                data["P/B Ratio"] = 0 # Screener usually requires calculation or login for P/B sometimes
                data["ROE"] = get_val("ROE")
                data["Book Value"] = get_val("Book Value")
                data["Div Yield"] = get_val("Dividend Yield")
                data["Industry"] = "Indian Equity"
                data["Source"] = "Screener.in"
                return data
        except: pass
        
    return None

def get_currency_symbol(ticker):
    if ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^"): return "₹"
    return "$"

# --------------------------
# 5. AI LOGIC (Hybrid)
def calculate_features(df):
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: x if x>0 else 0).rolling(14).mean() / df['Close'].diff().apply(lambda x: -x if x<0 else 0).rolling(14).mean()))
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df.dropna(inplace=True)
    return df

def run_hybrid_prediction(ticker):
    # Technicals
    df = robust_yf_download(ticker, "2y")
    if df is None or len(df) < 100: return None
    
    df = calculate_features(df)
    features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'ATR']
    
    # Train Model (Simplified Random Forest)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = df[features].iloc[:-1].values
    y = df['Target'].iloc[:-1].values
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Predict
    last_data = df[features].iloc[-1].values.reshape(1, -1)
    tech_signal = model.predict(last_data)[0]
    confidence = model.predict_proba(last_data)[0].max()
    
    # Fundamentals
    fund = get_fundamentals(ticker)
    fund_score = 0
    if fund:
        if 0 < fund['P/E Ratio'] < 25: fund_score += 1
        if fund['ROE'] > 15: fund_score += 1
        if fund['Div Yield'] > 1: fund_score += 1
    
    # Combined Verdict
    verdict = "HOLD"
    color = "white"
    
    if tech_signal == 1: # Buy Signal
        if fund_score >= 2: 
            verdict = "STRONG BUY 🚀"
            color = "#00d09c"
        else: 
            verdict = "SPECULATIVE BUY ⚠️"
            color = "#ffd700"
    else: # Sell Signal
        verdict = "SELL / AVOID 🔻"
        color = "#ff4b4b"

    # Targets
    curr = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    sl = curr - (1.5 * atr) if tech_signal == 1 else curr + (1.5 * atr)
    tgt = curr + (2.0 * atr) if tech_signal == 1 else curr - (2.0 * atr)
    
    return {'verdict':verdict, 'color':color, 'conf':confidence, 'sl':sl, 'tgt':tgt, 'curr':curr, 'fund':fund}

# --------------------------
# 6. APP STRUCTURE
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "SBIN.NS", "TATAMOTORS.NS", "ZOMATO.NS"]
ETF_LIST = {"Nifty BeES": "NIFTYBEES.NS", "Gold BeES": "GOLDBEES.NS", "Silver BeES": "SILVERBEES.NS", "Bank BeES": "BANKBEES.NS"}
COMMODITY_LIST = {"Gold ($)": "GC=F", "Silver ($)": "SI=F", "Crude Oil ($)": "CL=F", "Natural Gas ($)": "NG=F"}

st.sidebar.title("🚀 Menu")
nav = st.sidebar.radio("Go to", ["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"], index=["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"].index(st.session_state.page))
if nav != st.session_state.page: navigate_to(nav)

if st.session_state.page != "⭐ Top 5 AI Picks": st_autorefresh(interval=30000, key="refresh")

# --------------------------
# VIEWS
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
    try:
        d = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
        for e in d.entries[:4]:
            st.markdown(f"<div class='news-box'><a href='{e.link}' target='_blank' style='color:white;text-decoration:none'>{e.title}</a></div>", unsafe_allow_html=True)
    except: st.error("News unavailable")

elif st.session_state.page == "📈 Stock Analyzer":
    st.markdown("### 📈 Pro Analyzer")
    if st.button("← Back"): navigate_to("🏠 Market Dashboard")
    
    c1, c2 = st.columns([1,3])
    ex = c1.selectbox("Market", ["NSE","BSE","Global"])
    sym = c2.text_input("Ticker", st.session_state.selected_ticker.replace(".NS","").replace(".BO",""))
    
    if ex == "NSE": full = f"{sym}.NS"
    elif ex == "BSE": full = f"{sym}.BO"
    else: full = sym 
    curr_sym = get_currency_symbol(full)
    
    with st.spinner("🤖 Running AI & Fetching Fundamentals..."):
        data = run_hybrid_prediction(full)
        
    if data:
        # AI CARD
        st.markdown(f"""
        <div class="fun-card" style="border-left: 8px solid {data['color']}; margin-bottom: 20px;">
            <h1 style="color:{data['color']}; margin:0;">{data['verdict']}</h1>
            <p style="color:#aaa;">Tech Confidence: {data['conf']*100:.0f}%</p>
            <hr style="border-color:#333;">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                <div><span>Entry</span><h3>{curr_sym}{data['curr']:.2f}</h3></div>
                <div><span>Stop Loss</span><h3 style="color:#ff4b4b;">{curr_sym}{data['sl']:.2f}</h3></div>
                <div><span>Target</span><h3 style="color:#00d09c;">{curr_sym}{data['tgt']:.2f}</h3></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # FUNDAMENTALS GRID
        st.markdown("#### 📊 Fundamental Health")
        fd = data['fund']
        if fd:
            f1, f2, f3, f4 = st.columns(4)
            f1.markdown(f"<div class='metric-container'><div class='metric-label'>Market Cap</div><div class='metric-value'>{curr_sym}{fd['Market Cap']/10000000:.0f} Cr</div></div>", unsafe_allow_html=True)
            f2.markdown(f"<div class='metric-container'><div class='metric-label'>P/E Ratio</div><div class='metric-value'>{fd['P/E Ratio']:.2f}</div></div>", unsafe_allow_html=True)
            f3.markdown(f"<div class='metric-container'><div class='metric-label'>ROE</div><div class='metric-value'>{fd['ROE']:.2f}%</div></div>", unsafe_allow_html=True)
            f4.markdown(f"<div class='metric-container'><div class='metric-label'>Book Value</div><div class='metric-value'>{curr_sym}{fd['Book Value']:.2f}</div></div>", unsafe_allow_html=True)
            st.caption(f"Data Source: {fd['Source']}")
        else:
            st.warning("Fundamental data could not be fetched (Tried Yahoo & Screener).")

        # CHART
        df_chart = robust_yf_download(full, "1y")
        fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'])])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig, use_container_width=True)

    else: st.error("Data not found. Check ticker symbol.")

elif st.session_state.page == "📊 F/O Dashboard":
    st.markdown("### 📊 Option Chain (NSE Live)")
    fo_sym = st.text_input("Symbol", "NIFTY")
    data = get_nse_chain(fo_sym)
    if data:
        try:
            recs = data['records']['data']
            sel_exp = st.selectbox("Expiry", data['records']['expiryDates'])
            chain = [x for x in recs if x['expiryDate'] == sel_exp]
            spot = data['records']['underlyingValue']
            st.metric(f"{fo_sym} Spot", f"₹{spot:,.2f}")
            
            ce_oi = sum([x['CE']['openInterest'] for x in chain if 'CE' in x])
            pe_oi = sum([x['PE']['openInterest'] for x in chain if 'PE' in x])
            pcr = pe_oi / ce_oi if ce_oi > 0 else 0
            st.info(f"PCR: {pcr:.2f} | Sentiment: {'BULLISH' if pcr>1 else 'BEARISH'}")
            
            rows = []
            for x in chain:
                if x['strikePrice'] > spot*0.98 and x['strikePrice'] < spot*1.02:
                    row = {'Strike': x['strikePrice']}
                    if 'CE' in x: row.update({'CE Price': x['CE']['lastPrice'], 'CE OI': x['CE']['openInterest']})
                    if 'PE' in x: row.update({'PE Price': x['PE']['lastPrice'], 'PE OI': x['PE']['openInterest']})
                    rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index('Strike'), use_container_width=True)
        except: st.error("Data Parsing Error.")
    else: st.error("NSE Connection Failed (Use Localhost).")

elif st.session_state.page == "🏦 ETFs & Commodities":
    st.markdown("### 🏦 ETFs & Commodities")
    type_ = st.radio("Asset", ["ETFs (India)", "Commodities (Global)"], horizontal=True)
    active_list = ETF_LIST if type_ == "ETFs (India)" else COMMODITY_LIST
    cols = st.columns(4)
    for i, (name, ticker) in enumerate(active_list.items()):
        with cols[i % 4]:
            if st.button(name, key=name): navigate_to("📈 Stock Analyzer", ticker)

elif st.session_state.page == "⭐ Top 5 AI Picks":
    st.markdown("### ⭐ AI Scanners")
    if st.button("Start Scan"):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(SCANNER_POOL[:6]):
            bar.progress((i+1)/6)
            res = run_hybrid_prediction(t)
            if res: results.append((t, res['verdict'], res['curr'], res['color']))
        bar.empty()
        for t, ver, curr, clr in results:
            st.markdown(f"<div class='fun-card' style='border-left:5px solid {clr}'><b>{t}</b>: {ver} @ ₹{curr:.2f}</div>", unsafe_allow_html=True)
