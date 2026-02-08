import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
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
# 5. FUNDAMENTAL ANALYSIS ENGINE (NEW)
@st.cache_data(ttl=3600) # Cache for 1 hour since fundamentals don't change fast
def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract Key Metrics
        data = {
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "P/B Ratio": info.get("priceToBook", 0),
            "ROE": info.get("returnOnEquity", 0),
            "Book Value": info.get("bookValue", 0),
            "Div Yield": info.get("dividendYield", 0),
            "Industry": info.get("industry", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Current Price": info.get("currentPrice", 0),
            "52W High": info.get("fiftyTwoWeekHigh", 0)
        }
        return data
    except:
        return None

def calculate_fundamental_score(data):
    if not data: return 0, "N/A"
    
    score = 0
    reasons = []
    
    # 1. P/E Scoring (Simplified Valuation)
    pe = data["P/E Ratio"]
    if pe > 0 and pe < 20: 
        score += 2
        reasons.append("Undervalued (Low P/E)")
    elif pe >= 20 and pe < 40: 
        score += 1
    else: 
        reasons.append("Overvalued (High P/E)")

    # 2. ROE Scoring (Profitability)
    roe = data["ROE"]
    if roe > 0.20: 
        score += 2
        reasons.append("High Profitability (ROE > 20%)")
    elif roe > 0.10: 
        score += 1
    
    # 3. P/B Scoring (Value)
    pb = data["P/B Ratio"]
    if pb > 0 and pb < 3: 
        score += 1
        reasons.append("Good Value (P/B < 3)")
        
    return score, ", ".join(reasons)

# --------------------------
# 6. TECHNICAL AI ENGINE
def calculate_features(df):
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: x if x>0 else 0).rolling(14).mean() / df['Close'].diff().apply(lambda x: -x if x<0 else 0).rolling(14).mean()))
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df.dropna(inplace=True)
    return df

def run_hybrid_prediction(ticker):
    # 1. Technical Analysis
    df = robust_yf_download(ticker, "2y")
    if df is None or len(df) < 100: return None
    
    df = calculate_features(df)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'ATR']
    X = df[features].iloc[:-1].values
    y = df['Target'].iloc[:-1].values
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    model.fit(X, y)
    
    last_data = df[features].iloc[-1].values.reshape(1, -1)
    tech_signal = model.predict(last_data)[0] # 1=Buy, 0=Sell
    confidence = model.predict_proba(last_data)[0].max()
    
    # 2. Fundamental Analysis
    fund_data = get_fundamentals(ticker)
    fund_score, fund_reason = calculate_fundamental_score(fund_data)
    
    # 3. Hybrid Verdict
    final_verdict = "HOLD"
    color = "white"
    
    if tech_signal == 1: # Technical BUY
        if fund_score >= 3:
            final_verdict = "STRONG BUY 🚀"
            color = "#00d09c"
        else:
            final_verdict = "CAUTIOUS BUY ⚠️"
            color = "#ffd700" # Gold
    else: # Technical SELL
        if fund_score >= 4:
            final_verdict = "ACCUMULATE (DIP) 🛒"
            color = "#00bfff" # Blue
        else:
            final_verdict = "STRONG SELL 🔻"
            color = "#ff4b4b"

    curr = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    sl = curr - (1.5 * atr) if tech_signal == 1 else curr + (1.5 * atr)
    tgt = curr + (2.0 * atr) if tech_signal == 1 else curr - (2.0 * atr)
    
    return {
        'verdict': final_verdict,
        'color': color,
        'fund_data': fund_data,
        'fund_score': fund_score,
        'tech_conf': confidence,
        'sl': sl,
        'tgt': tgt,
        'curr': curr
    }

# --------------------------
# 7. DATA LISTS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "SBIN.NS", "TATAMOTORS.NS", "ZOMATO.NS"]
ETF_LIST = {"Nifty BeES": "NIFTYBEES.NS", "Gold BeES": "GOLDBEES.NS", "Silver BeES": "SILVERBEES.NS", "Bank BeES": "BANKBEES.NS"}
COMMODITY_LIST = {"Gold ($)": "GC=F", "Silver ($)": "SI=F", "Crude Oil ($)": "CL=F", "Natural Gas": "NG=F"}

# --------------------------
# 8. SIDEBAR
st.sidebar.title("🚀 Menu")
nav = st.sidebar.radio("Go to", ["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"], index=["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "🏦 ETFs & Commodities", "⭐ Top 5 AI Picks"].index(st.session_state.page))
if nav != st.session_state.page: navigate_to(nav)

if st.session_state.page != "⭐ Top 5 AI Picks":
    st_autorefresh(interval=30000, key="refresh")

# --------------------------
# 9. DASHBOARD
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
# 10. STOCK ANALYZER (HYBRID AI)
elif st.session_state.page == "📈 Stock Analyzer":
    st.markdown("### 📈 Pro Analyzer (Tech + Fundamentals)")
    if st.button("← Back"): navigate_to("🏠 Market Dashboard")
    
    c1, c2 = st.columns([1,3])
    ex = c1.selectbox("Market", ["NSE","BSE","Global"])
    clean_default = st.session_state.selected_ticker.replace(".NS","").replace(".BO","")
    sym = c2.text_input("Ticker", clean_default)
    
    if ex == "NSE": full = f"{sym}.NS"
    elif ex == "BSE": full = f"{sym}.BO"
    else: full = sym 
    curr_sym = get_currency_symbol(full)
    
    # 1. Run Analysis
    with st.spinner("🤖 Running Hybrid AI (Fundamentals + Technicals)..."):
        data = run_hybrid_prediction(full)
        
    if data:
        # 2. AI VERDICT CARD
        st.markdown(f"""
        <div class="fun-card" style="border-left: 8px solid {data['color']}; margin-bottom: 20px;">
            <h1 style="color:{data['color']}; margin:0;">{data['verdict']}</h1>
            <p style="color:#aaa;">Confidence: {data['tech_conf']*100:.0f}% | Fundamental Score: {data['fund_score']}/5</p>
            <hr style="border-color:#333;">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                <div><span>Entry</span><h3>{curr_sym}{data['curr']:.2f}</h3></div>
                <div><span>Stop Loss</span><h3 style="color:#ff4b4b;">{curr_sym}{data['sl']:.2f}</h3></div>
                <div><span>Target</span><h3 style="color:#00d09c;">{curr_sym}{data['tgt']:.2f}</h3></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. FUNDAMENTALS GRID
        st.markdown("#### 📊 Fundamental Health")
        fd = data['fund_data']
        if fd:
            f1, f2, f3, f4 = st.columns(4)
            f1.markdown(f"<div class='metric-container'><div class='metric-label'>Market Cap</div><div class='metric-value'>{curr_sym}{fd['Market Cap']/10000000:.0f} Cr</div></div>", unsafe_allow_html=True)
            f2.markdown(f"<div class='metric-container'><div class='metric-label'>P/E Ratio</div><div class='metric-value'>{fd['P/E Ratio']:.2f}</div></div>", unsafe_allow_html=True)
            f3.markdown(f"<div class='metric-container'><div class='metric-label'>ROE</div><div class='metric-value'>{fd['ROE']*100:.1f}%</div></div>", unsafe_allow_html=True)
            f4.markdown(f"<div class='metric-container'><div class='metric-label'>P/B Ratio</div><div class='metric-value'>{fd['P/B Ratio']:.2f}</div></div>", unsafe_allow_html=True)
            
            f5, f6, f7, f8 = st.columns(4)
            f5.markdown(f"<div class='metric-container'><div class='metric-label'>Book Val</div><div class='metric-value'>{curr_sym}{fd['Book Value']:.2f}</div></div>", unsafe_allow_html=True)
            f6.markdown(f"<div class='metric-container'><div class='metric-label'>Div Yield</div><div class='metric-value'>{fd['Div Yield']*100:.2f}%</div></div>", unsafe_allow_html=True)
            f7.markdown(f"<div class='metric-container'><div class='metric-label'>Sector</div><div class='metric-value' style='font-size:12px;'>{fd['Sector']}</div></div>", unsafe_allow_html=True)
            f8.markdown(f"<div class='metric-container'><div class='metric-label'>Industry</div><div class='metric-value' style='font-size:12px;'>{fd['Industry'][:15]}...</div></div>", unsafe_allow_html=True)
        else:
            st.warning("Fundamental data unavailable for this ticker.")

        # 4. CHART
        st.markdown("#### 📉 Price Action")
        df_chart = robust_yf_download(full, "1y")
        fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'])])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig, use_container_width=True)

    else: st.error("Data not found.")

# --------------------------
# 11. F/O DASHBOARD
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
            
            rows = []
            for x in chain:
                if x['strikePrice'] > spot*0.98 and x['strikePrice'] < spot*1.02:
                    row = {'Strike': x['strikePrice']}
                    if 'CE' in x: row.update({'CE Price': x['CE']['lastPrice'], 'CE OI': x['CE']['openInterest']})
                    if 'PE' in x: row.update({'PE Price': x['PE']['lastPrice'], 'PE OI': x['PE']['openInterest']})
                    rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index('Strike'), use_container_width=True)
        except: st.error("Data Parse Error")
    else: st.error("NSE Connection Failed (Try Localhost).")

# --------------------------
# 12. ETF & COMMODITIES
elif st.session_state.page == "🏦 ETFs & Commodities":
    st.markdown("### 🏦 ETFs & Commodities")
    type_ = st.radio("Asset", ["ETFs (India)", "Commodities (Global)"], horizontal=True)
    active_list = ETF_LIST if type_ == "ETFs (India)" else COMMODITY_LIST
    cols = st.columns(4)
    for i, (name, ticker) in enumerate(active_list.items()):
        with cols[i % 4]:
            if st.button(name, key=name): navigate_to("📈 Stock Analyzer", ticker)

# --------------------------
# 13. AI PICKS
elif st.session_state.page == "⭐ Top 5 AI Picks":
    st.markdown("### ⭐ AI Scanners")
    if st.button("Start Scan"):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(SCANNER_POOL[:6]):
            bar.progress((i+1)/6)
            res = run_hybrid_prediction(t) # Uses Hybrid Logic
            if res: results.append((t, res['verdict'], res['curr'], res['color']))
        bar.empty()
        for t, ver, curr, clr in results:
            st.markdown(f"<div class='fun-card' style='border-left:5px solid {clr}'><b>{t}</b>: {ver} @ ₹{curr:.2f}</div>", unsafe_allow_html=True)
            
