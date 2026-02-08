import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import requests
import feedparser
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# --------------------------
# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Groww-Style Pro AI", 
    layout="wide", 
    page_icon="🚀",
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
        background: rgba(30, 34, 45, 0.6); 
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; 
        padding: 16px; 
        cursor: pointer; 
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fun-card:hover { transform: translateY(-3px); border-color: #00d09c; }
    
    /* NEWS */
    .news-box { background: #161920; padding: 12px; border-radius: 10px; margin-bottom: 8px; border-left: 3px solid #4c8bf5; }
    .news-link { color: #fff; text-decoration: none; font-size: 14px; }
    
    /* BUTTONS */
    div.stButton > button { width: 100%; background: transparent; border: none; color: white; text-align: left; padding: 0; }
    div.stButton > button:hover { color: #00d09c; background: transparent; }
    
    /* PREDICTION CARD */
    .pred-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #333;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. ROBUST DATA ENGINE
def get_smart_interval(period):
    if period in ["1d", "5d"]: return "1m"
    if period == "1mo": return "30m"
    if period == "3mo": return "60m"
    if period in ["6mo", "1y"]: return "1d"
    return "1wk"

def robust_yf_download(ticker, period):
    try:
        interval = get_smart_interval(period)
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        return df
    except: return None

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

def get_currency_symbol(ticker):
    if ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^"): return "₹"
    return "$"

# --------------------------
# 5. ADVANCED ML PREDICTOR
def calculate_features(df):
    df = df.copy()
    # Indicators
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: x if x>0 else 0).rolling(14).mean() / df['Close'].diff().apply(lambda x: -x if x<0 else 0).rolling(14).mean()))
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df.dropna(inplace=True)
    return df

def run_ml_prediction(ticker):
    # 1. Fetch Data
    df = robust_yf_download(ticker, "2y")
    if df is None or len(df) < 100: return None
    
    # 2. Prep Features
    df = calculate_features(df)
    
    # 3. Create Target (Next Day Up/Down)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # 4. Train/Test Split
    features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'ATR']
    X = df[features].iloc[:-1].values
    y = df['Target'].iloc[:-1].values
    
    if len(X) < 50: return None
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    model.fit(X, y)
    
    # 5. Predict Next Move
    last_data = df[features].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(last_data)[0]
    prob = model.predict_proba(last_data)[0]
    
    confidence = prob[1] if prediction == 1 else prob[0]
    
    # 6. Calculate Levels
    curr = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    signal = "BUY" if prediction == 1 else "SELL"
    sl = curr - (1.5 * atr) if signal == "BUY" else curr + (1.5 * atr)
    tgt = curr + (2.0 * atr) if signal == "BUY" else curr - (2.0 * atr)
    
    return {
        'signal': signal,
        'conf': confidence,
        'sl': sl,
        'tgt': tgt,
        'curr': curr
    }

# --------------------------
# 6. DATA LISTS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "SBIN.NS", "TATAMOTORS.NS", "ZOMATO.NS"]
ETF_LIST = {"Nifty BeES": "NIFTYBEES.NS", "Gold BeES": "GOLDBEES.NS", "Silver BeES": "SILVERBEES.NS", "Bank BeES": "BANKBEES.NS"}
COMMODITY_LIST = {"Gold ($)": "GC=F", "Silver ($)": "SI=F", "Crude Oil ($)": "CL=F", "Natural Gas": "NG=F"}

# --------------------------
# 7. SIDEBAR
st.sidebar.title("🚀 Menu")
nav = st.sidebar.radio("Go to", ["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"], index=["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"].index(st.session_state.page))
if nav != st.session_state.page: navigate_to(nav)

if st.session_state.page != "⭐ Top 5 AI Picks":
    st_autorefresh(interval=30000, key="refresh")

# --------------------------
# 8. DASHBOARD
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

# --------------------------
# 9. STOCK ANALYZER (WITH ML)
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
    
    # 1. Timeframe
    tf_tabs = st.tabs(["1D", "5D", "1M", "6M", "1Y", "MAX"])
    tf_map = {"1D":"1d", "5D":"5d", "1M":"1mo", "6M":"6mo", "1Y":"1y", "MAX":"max"}
    sel_tf = st.radio("Timeframe", list(tf_map.keys()), horizontal=True, label_visibility="collapsed")
    period = tf_map[sel_tf]
    
    df = robust_yf_download(full, period)
    
    if df is not None:
        # Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. ML PREDICTION CARD
        with st.spinner("🤖 Running Random Forest Model..."):
            res = run_ml_prediction(full)
        
        if res:
            clr = "#00d09c" if res['signal']=="BUY" else "#ff4b4b"
            st.markdown(f"""
            <div class="pred-card" style="background:#1e2330; border-left:5px solid {clr};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2 style="color:{clr}; margin:0;">{res['signal']} <span style="font-size:16px; color:#aaa;">({res['conf']*100:.0f}% Conf)</span></h2>
                    <h3 style="margin:0;">{curr_sym}{res['curr']:.2f}</h3>
                </div>
                <hr style="border-color:#333;">
                <div style="display:flex; gap:30px;">
                    <div><span style="color:#ff4b4b;">STOP LOSS</span><br><b style="font-size:18px;">{curr_sym}{res['sl']:.2f}</b></div>
                    <div><span style="color:#00d09c;">TARGET</span><br><b style="font-size:18px;">{curr_sym}{res['tgt']:.2f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. STOCKTWITS
        clean_twit = sym.replace(".NS","")
        components.html(f"""<script type="text/javascript" src="https://api.stocktwits.com/addon/widget/2/widget-loader.min.js"></script><div id="stocktwits-widget-news"></div><script type="text/javascript">STWT.Widget({{container: 'stocktwits-widget-news', symbol: '{clean_twit}', width: '100%', height: '300', limit: '15', scrollbars: 'true', streaming: 'true', title: '{clean_twit} Stream', style: {{link_color: '48515c', link_hover_color: '48515c', header_text_color: 'ffffff', border_color: '333333', divider_color: '333333', box_color: '161920', stream_color: '161920', text_color: 'ffffff', time_color: '999999'}} }});</script>""", height=320, scrolling=True)

    else: st.error(f"No data found for {full}.")

# --------------------------
# 10. ETF & COMMODITIES
elif st.session_state.page == "🏦 ETFs & Commodities":
    st.markdown("### 🏦 ETFs & Commodities")
    type_ = st.radio("Asset", ["ETFs (India)", "Commodities (Global)"], horizontal=True)
    active_list = ETF_LIST if type_ == "ETFs (India)" else COMMODITY_LIST
    cols = st.columns(4)
    for i, (name, ticker) in enumerate(active_list.items()):
        with cols[i % 4]:
            if st.button(name, key=name): navigate_to("📈 Stock Analyzer", ticker)

# --------------------------
# 11. F/O DASHBOARD (NSE NATIVE)
elif st.session_state.page == "📊 F/O Dashboard":
    st.markdown("### 📊 Option Chain")
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
            st.info(f"PCR: {pcr:.2f} | Total OI: {ce_oi+pe_oi:,.0f}")
            
            rows = []
            for x in chain:
                if x['strikePrice'] > spot*0.98 and x['strikePrice'] < spot*1.02:
                    row = {'Strike': x['strikePrice']}
                    if 'CE' in x: row.update({'CE Price': x['CE']['lastPrice'], 'CE OI': x['CE']['openInterest']})
                    if 'PE' in x: row.update({'PE Price': x['PE']['lastPrice'], 'PE OI': x['PE']['openInterest']})
                    rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index('Strike'), use_container_width=True)
        except: st.error("Data Parse Error")
    else: st.error("NSE Connection Failed (Try Localhost)")

# --------------------------
# 12. AI PICKS
elif st.session_state.page == "⭐ Top 5 AI Picks":
    st.markdown("### ⭐ Random Forest Scanners")
    if st.button("Start Scan"):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(SCANNER_POOL[:6]):
            bar.progress((i+1)/6)
            res = run_ml_prediction(t)
            if res: results.append((t, res['signal'], res['curr'], res['conf']))
        bar.empty()
        for t, sig, curr, conf in results:
            clr = "#00d09c" if sig=="BUY" else "#ff4b4b"
            st.markdown(f"<div class='fun-card' style='border-left:5px solid {clr}'><b>{t}</b>: {sig} ({conf*100:.0f}%) @ ₹{curr:.2f}</div>", unsafe_allow_html=True)
