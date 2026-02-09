import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import requests
import feedparser
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from textblob import TextBlob

# --------------------------
# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Pro Market Dashboard V12", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# --------------------------
# 2. SESSION STATE
if 'page' not in st.session_state: st.session_state.page = "🏠 Market Dashboard"
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = "FORCEMOT.NS"

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
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; 
        padding: 15px; 
        cursor: pointer; 
        transition: 0.3s;
    }
    .fun-card:hover { transform: translateY(-3px); border-color: #00d09c; }
    
    /* NEWS */
    .news-box { background: #161920; padding: 12px; border-radius: 10px; margin-bottom: 8px; border-left: 3px solid #4c8bf5; }
    
    /* BUTTONS */
    div.stButton > button { width: 100%; background: transparent; border: none; color: white; text-align: left; padding: 0; }
    div.stButton > button:hover { color: #00d09c; background: transparent; }
    
    /* METRICS */
    .metric-container { background: #1e2330; padding: 10px; border-radius: 8px; border: 1px solid #333; text-align: center; margin-bottom: 5px; }
    .metric-label { font-size: 10px; color: #aaa; text-transform: uppercase; }
    .metric-value { font-size: 14px; font-weight: bold; color: #fff; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. DATA ENGINES

def robust_yf_download(ticker, period):
    try:
        interval = "1m" if period in ["1d","5d"] else "1d"
        if period == "1mo": interval = "30m"
        
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Fallback if intraday is empty
        if df.empty and interval != "1d":
            df = yf.download(ticker, period=period, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
        return df if not df.empty else None
    except: return None

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    # Try Yahoo First
    try:
        info = yf.Ticker(ticker).info
        if info.get('trailingPE') is not None:
            return {
                "Market Cap": info.get("marketCap", 0),
                "P/E": info.get("trailingPE", 0),
                "P/B": info.get("priceToBook", 0),
                "ROE": (info.get("returnOnEquity", 0) or 0) * 100,
                "Book Value": info.get("bookValue", 0),
                "52W High": info.get("fiftyTwoWeekHigh", 0),
                "Source": "Yahoo"
            }
    except: pass
    
    # Fallback Screener
    if ".NS" in ticker:
        try:
            url = f"https://www.screener.in/company/{ticker.replace('.NS','')}/"
            r = requests.get(url)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                lis = soup.find('ul', {'id': 'top-ratios'}).find_all('li')
                d = {}
                for li in lis:
                    name = li.find('span', {'class':'name'}).text.strip()
                    val = li.find('span', {'class':'number'}).text.replace(',','')
                    if "Market Cap" in name: d["Market Cap"] = float(val)*10000000
                    if "Stock P/E" in name: d["P/E"] = float(val)
                    if "ROE" in name: d["ROE"] = float(val)
                    if "Book Value" in name: d["Book Value"] = float(val)
                    if "High / Low" in name: d["52W High"] = float(val.split('/')[0].strip())
                d["Source"] = "Screener"
                return d
        except: pass
    return None

def get_news_sentiment(ticker):
    try:
        clean = ticker.replace(".NS","").replace(".BO","")
        url = f"https://news.google.com/rss/search?q={clean}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        sentiment_score = 0
        for e in feed.entries[:5]:
            analysis = TextBlob(e.title)
            sentiment_score += analysis.sentiment.polarity
        return sentiment_score # > 0 is Bullish, < 0 is Bearish
    except: return 0

# --------------------------
# 5. SMART AI ENGINE (V12.0 Logic)

def calculate_features(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: x if x>0 else 0).rolling(14).mean() / df['Close'].diff().apply(lambda x: -x if x<0 else 0).rolling(14).mean()))
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['Vol_SMA'] = df['Volume'].rolling(20).mean() # Volume Average
    df.dropna(inplace=True)
    return df

def run_smart_prediction(ticker):
    # 1. Fetch Data
    df = robust_yf_download(ticker, "2y")
    if df is None or len(df) < 50: return None
    
    df = calculate_features(df)
    
    # 2. Extract Key Metrics
    curr_price = df['Close'].iloc[-1]
    curr_vol = df['Volume'].iloc[-1]
    avg_vol = df['Vol_SMA'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    
    # 3. Get Fundamentals & Sentiment
    fund = get_fundamentals(ticker)
    sent_score = get_news_sentiment(ticker)
    
    # --- THE V12.0 LOGIC CORE ---
    verdict = "HOLD"
    color = "white"
    reason = "Market is choppy."
    
    # Condition A: MOMENTUM BREAKOUT (The Force Motors Fix)
    is_breakout = False
    if curr_vol > 1.5 * avg_vol and curr_price > sma_50:
        is_breakout = True
        
    # Condition B: 52-Week High Breakout
    near_52w_high = False
    if fund and fund.get('52W High', 0) > 0:
        if curr_price >= fund['52W High'] * 0.98: # Within 2% of ATH
            near_52w_high = True

    # --- DECISION TREE ---
    if is_breakout or near_52w_high:
        verdict = "MOMENTUM BUY 🚀"
        color = "#00d09c"
        reason = "High Volume Breakout / Near ATH. Trend is very strong."
    
    elif rsi < 30:
        verdict = "OVERSOLD BUY 🟢"
        color = "#00d09c"
        reason = "RSI is low (<30). Good dip buying opportunity."
        
    elif rsi > 75 and not is_breakout:
        verdict = "OVERBOUGHT SELL 🔻"
        color = "#ff4b4b"
        reason = "RSI is too high (>75) without volume support. Profit booking likely."
        
    elif curr_price > sma_50 and sent_score >= 0:
        verdict = "BUY (TREND) 📈"
        color = "#00d09c"
        reason = "Price above 50 SMA and Sentiment is Neutral/Positive."
        
    elif curr_price < sma_50:
        verdict = "SELL (WEAK) 📉"
        color = "#ff4b4b"
        reason = "Price below 50 SMA. Trend is bearish."

    # 4. Fundamental Check
    if fund:
        pe = fund.get('P/E', 0)
        if pe > 80 and "BUY" in verdict and not is_breakout:
            verdict = "CAUTIOUS BUY ⚠️"
            color = "#ffd700"
            reason += " But Valuation (P/E) is very high."

    # 5. Targets
    sl = curr_price - (2.0 * atr) if "BUY" in verdict else curr_price + (1.5 * atr)
    tgt = curr_price + (3.0 * atr) if "BUY" in verdict else curr_price - (2.0 * atr)
    
    return {
        'verdict': verdict, 'color': color, 'reason': reason,
        'sl': sl, 'tgt': tgt, 'curr': curr_price, 
        'fund': fund, 'sent': sent_score
    }

# --------------------------
# 6. APP STRUCTURE
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["FORCEMOT.NS", "POWERINDIA.NS" , "RELIANCE.NS", "ZOMATO.NS", "TATASTEEL.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS"] 

st.sidebar.title("🚀 Menu")
nav = st.sidebar.radio("Go to", ["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "⭐ Top 5 AI Picks"], index=["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "⭐ Top 5 AI Picks"].index(st.session_state.page))
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
            chg = curr - df['Close'].iloc[-2]
            pct = (chg/df['Close'].iloc[-2])*100
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
    ex = c1.selectbox("Market", ["NSE","BSE"])
    raw = st.session_state.selected_ticker.replace(".NS","").replace(".BO","")
    sym = c2.text_input("Ticker", raw)
    full = f"{sym.upper().strip()}.NS" if ex == "NSE" else f"{sym.upper().strip()}.BO"
    
    with st.spinner(f"🔍 Analyzing {full}..."):
        data = run_smart_prediction(full)
    
    if data:
        st.markdown(f"""
        <div class="fun-card" style="border-left: 8px solid {data['color']}; margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h1 style="color:{data['color']}; margin:0;">{data['verdict']}</h1>
                <div style="text-align:right;">
                    <h2 style="margin:0;">₹{data['curr']:,.2f}</h2>
                </div>
            </div>
            <p style="color:#ddd; margin-top:5px;"><i>Reason: {data['reason']}</i></p>
            <hr style="border-color:#333;">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                <div><span>🛑 Stop Loss</span><h3 style="color:#ff4b4b;">₹{data['sl']:.2f}</h3></div>
                <div><span>🎯 Target</span><h3 style="color:#00d09c;">₹{data['tgt']:.2f}</h3></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Vital Stats")
        fd = data['fund']
        if fd:
            f1, f2, f3, f4 = st.columns(4)
            f1.markdown(f"<div class='metric-container'><div class='metric-label'>Market Cap</div><div class='metric-value'>₹{fd['Market Cap']/10000000:.0f} Cr</div></div>", unsafe_allow_html=True)
            f2.markdown(f"<div class='metric-container'><div class='metric-label'>P/E Ratio</div><div class='metric-value'>{fd['P/E']:.2f}</div></div>", unsafe_allow_html=True)
            f3.markdown(f"<div class='metric-container'><div class='metric-label'>ROE</div><div class='metric-value'>{fd['ROE']:.1f}%</div></div>", unsafe_allow_html=True)
            f4.markdown(f"<div class='metric-container'><div class='metric-label'>Book Val</div><div class='metric-value'>₹{fd['Book Value']:.2f}</div></div>", unsafe_allow_html=True)
        else: st.warning("Fundamentals unavailable.")

        df_chart = robust_yf_download(full, "1y")
        fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'])])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig, use_container_width=True)

    else: st.error("Data not found.")

elif st.session_state.page == "📊 F/O Dashboard":
    st.markdown("### 📊 F/O Dashboard (NSE Live)")
    st.info("NSE Connection Logic Active.") # Placeholder for F/O logic

elif st.session_state.page == "⭐ Top 5 AI Picks":
    st.markdown("### ⭐ Smart Momentum Scanners")
    if st.button("Start Scan"):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(SCANNER_POOL[:8]):
            bar.progress((i+1)/8)
            res = run_smart_prediction(t)
            if res: results.append((t, res['verdict'], res['curr'], res['color']))
        bar.empty()
        for t, ver, curr, clr in results:
            st.markdown(f"<div class='fun-card' style='border-left:5px solid {clr}'><b>{t}</b>: {ver} @ ₹{curr:.2f}</div>", unsafe_allow_html=True)
