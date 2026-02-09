import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import feedparser
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# --------------------------
# 0. INITIAL SETUP
# --------------------------
# Download VADER lexicon for sentiment analysis if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

st.set_page_config(
    page_title="Quant Market Dashboard V13", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# --------------------------
# 1. SESSION STATE
# --------------------------
if 'page' not in st.session_state: st.session_state.page = "🏠 Market Dashboard"
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = "FORCEMOT.NS"

def navigate_to(page, ticker=None):
    st.session_state.page = page
    if ticker: st.session_state.selected_ticker = ticker
    st.rerun()

# --------------------------
# 2. MODERN UI CSS
# --------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f1115; font-family: 'Inter', sans-serif; }
    
    /* CARDS */
    .fun-card {
        background: rgba(30, 34, 45, 0.6); 
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; 
        padding: 15px; 
        cursor: pointer; 
        transition: 0.3s;
    }
    .fun-card:hover { transform: translateY(-3px); border-color: #00d09c; }
    
    /* SIGNAL BOX */
    .signal-box {
        background: #1e2330;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        border: 1px solid #333;
    }

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
# 3. ROBUST DATA ENGINES
# --------------------------

def robust_yf_download(ticker, period):
    try:
        # Smart Interval: 1m for short term, 1d for long term
        interval = "1m" if period in ["1d","5d"] else "1d"
        if period == "1mo": interval = "30m"
        
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Flatten MultiIndex Columns (Fix for yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Fallback if intraday fails
        if df.empty and interval != "1d":
            df = yf.download(ticker, period=period, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
        return df if not df.empty else None
    except: return None

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    # 1. Yahoo Finance
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
    
    # 2. Screener.in Fallback (For Indian Stocks)
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

def get_news_sentiment_vader(ticker):
    """Fetches news and scores it using VADER"""
    try:
        clean = ticker.replace(".NS","").replace(".BO","")
        url = f"https://news.google.com/rss/search?q={clean}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        scores = []
        for e in feed.entries[:5]:
            score = sia.polarity_scores(e.title)['compound']
            scores.append(score)
        return np.mean(scores) if scores else 0
    except: return 0

@st.cache_data(ttl=60)
def get_nse_chain(symbol="NIFTY"):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        clean_sym = symbol.upper().replace(".NS","").replace("^","")
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={clean_sym}" if clean_sym in ["NIFTY","BANKNIFTY"] else f"https://www.nseindia.com/api/option-chain-equities?symbol={clean_sym}"
        return session.get(url, headers=headers, timeout=5).json()
    except: return None

# --------------------------
# 4. ADVANCED QUANT LOGIC (YOUR NEW CODE INTEGRATED)
# --------------------------

def calculate_advanced_features(df):
    df = df.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # RSI (Wilder's Smoothing)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Volume & ATR
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA'].replace(0, np.nan)
    
    df.dropna(inplace=True)
    return df

def run_quant_prediction(ticker):
    # 1. Fetch Data
    df = robust_yf_download(ticker, "2y")
    if df is None or len(df) < 50: return None
    
    df = calculate_advanced_features(df)
    
    # 2. Key Metrics
    curr_price = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]
    vol_ratio = df['Vol_Ratio'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    # 3. Fundamentals & Sentiment
    fund = get_fundamentals(ticker)
    sent_score = get_news_sentiment_vader(ticker)
    
    # 4. LOGIC ENGINE
    verdict = "HOLD / NEUTRAL"
    color = "#888888"
    reason = "No clear directional signal."
    
    # Signal A: MOMENTUM BREAKOUT (Force Motors Style)
    if vol_ratio > 1.5 and curr_price > sma_50 and macd_hist > 0:
        verdict = "MOMENTUM BUY 🚀"
        color = "#00d09c"
        reason = "Volume Spike (>1.5x) + Price > 50SMA + MACD Bullish."
    
    # Signal B: OVERSOLD REVERSAL
    elif rsi < 30 and macd_hist > 0:
        verdict = "REVERSAL BUY 🟢"
        color = "#00d09c"
        reason = "RSI Oversold (<30) + MACD turning positive."
        
    # Signal C: TREND CONTINUATION
    elif curr_price > sma_50 and 50 < rsi < 75 and macd_hist > 0 and sent_score >= 0:
        verdict = "TREND BUY 📈"
        color = "#00d09c"
        reason = "Uptrend verified by MACD & Sentiment."
        
    # Signal D: OVERBOUGHT / WEAKNESS
    elif rsi > 75 and macd_hist < 0:
        verdict = "SELL (Overbought) 🔻"
        color = "#ff4b4b"
        reason = "RSI Overbought (>75) + MACD weakening."
        
    # Signal E: DOWNTREND
    elif curr_price < sma_50 and macd_hist < 0:
        verdict = "SELL (Downtrend) 📉"
        color = "#ff4b4b"
        reason = "Price < 50SMA + MACD Negative."

    # 5. Targets
    sl = curr_price - (2.0 * atr) if "BUY" in verdict else curr_price + (1.5 * atr)
    tgt = curr_price + (3.0 * atr) if "BUY" in verdict else curr_price - (2.0 * atr)
    
    return {
        'verdict': verdict, 'color': color, 'reason': reason,
        'sl': sl, 'tgt': tgt, 'curr': curr_price, 
        'fund': fund, 'sent': sent_score
    }

# --------------------------
# 5. APP STRUCTURE
# --------------------------
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["FORCEMOT.NS", "RELIANCE.NS", "ZOMATO.NS", "TATASTEEL.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS"] 

st.sidebar.title("🚀 Menu")
nav = st.sidebar.radio("Go to", ["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "⭐ Top 5 AI Picks"], index=["🏠 Market Dashboard", "📈 Stock Analyzer", "📊 F/O Dashboard", "⭐ Top 5 AI Picks"].index(st.session_state.page))
if nav != st.session_state.page: navigate_to(nav)

if st.session_state.page != "⭐ Top 5 AI Picks": st_autorefresh(interval=30000, key="refresh")

# --- DASHBOARD ---
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

# --- STOCK ANALYZER (QUANT) ---
elif st.session_state.page == "📈 Stock Analyzer":
    st.markdown("### 📈 Quant Analyzer")
    if st.button("← Back"): navigate_to("🏠 Market Dashboard")
    
    c1, c2 = st.columns([1,3])
    ex = c1.selectbox("Market", ["NSE","BSE"])
    raw = st.session_state.selected_ticker.replace(".NS","").replace(".BO","")
    sym = c2.text_input("Ticker", raw)
    full = f"{sym.upper().strip()}.NS" if ex == "NSE" else f"{sym.upper().strip()}.BO"
    
    with st.spinner(f"🔍 Running VADER Sentiment & Technicals on {full}..."):
        data = run_quant_prediction(full)
    
    if data:
        # 1. SIGNAL CARD
        st.markdown(f"""
        <div class="signal-box" style="border-left: 6px solid {data['color']};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h1 style="color:{data['color']}; margin:0;">{data['verdict']}</h1>
                <h2 style="margin:0;">₹{data['curr']:,.2f}</h2>
            </div>
            <p style="color:#ddd; margin-top:5px;"><i>Reason: {data['reason']}</i></p>
            <hr style="border-color:#333;">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                <div><span>🛑 Stop Loss</span><h3 style="color:#ff4b4b;">₹{data['sl']:.2f}</h3></div>
                <div><span>🎯 Target</span><h3 style="color:#00d09c;">₹{data['tgt']:.2f}</h3></div>
                <div><span>📰 Sentiment</span><h3>{data['sent']:.2f}</h3></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. FUNDAMENTALS
        st.markdown("#### 📊 Vital Stats")
        fd = data['fund']
        if fd:
            f1, f2, f3, f4 = st.columns(4)
            f1.markdown(f"<div class='metric-container'><div class='metric-label'>Market Cap</div><div class='metric-value'>₹{fd['Market Cap']/10000000:.0f} Cr</div></div>", unsafe_allow_html=True)
            f2.markdown(f"<div class='metric-container'><div class='metric-label'>P/E Ratio</div><div class='metric-value'>{fd['P/E']:.2f}</div></div>", unsafe_allow_html=True)
            f3.markdown(f"<div class='metric-container'><div class='metric-label'>ROE</div><div class='metric-value'>{fd['ROE']:.1f}%</div></div>", unsafe_allow_html=True)
            f4.markdown(f"<div class='metric-container'><div class='metric-label'>Book Val</div><div class='metric-value'>₹{fd['Book Value']:.2f}</div></div>", unsafe_allow_html=True)
        else: st.warning("Fundamentals unavailable via API.")

        # 3. CHART
        df_chart = robust_yf_download(full, "1y")
        fig = go.Figure(data=[go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'])])
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig, use_container_width=True)

    else: st.error("Data not found. Check ticker symbol.")

# --- F/O DASHBOARD ---
elif st.session_state.page == "📊 F/O Dashboard":
    st.markdown("### 📊 F/O Dashboard (NSE Live)")
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
            st.info(f"PCR: {pcr:.2f} | Total OI: {ce_oi+pe_oi:,.0f}")
            
            rows = []
            for x in chain:
                if x['strikePrice'] > spot*0.98 and x['strikePrice'] < spot*1.02:
                    row = {'Strike': x['strikePrice']}
                    if 'CE' in x: row.update({'CE Price': x['CE']['lastPrice'], 'CE OI': x['CE']['openInterest']})
                    if 'PE' in x: row.update({'PE Price': x['PE']['lastPrice'], 'PE OI': x['PE']['openInterest']})
                    rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index('Strike'), use_container_width=True)
        except: st.error("Data Parsing Error.")
    else: st.error("NSE Connection Failed (Try Localhost).")

# --- SCANNER ---
elif st.session_state.page == "⭐ Top 5 AI Picks":
    st.markdown("### ⭐ Smart Momentum Scanners")
    if st.button("Start Scan"):
        results = []
        bar = st.progress(0)
        for i, t in enumerate(SCANNER_POOL[:8]):
            bar.progress((i+1)/8)
            res = run_quant_prediction(t)
            if res: results.append((t, res['verdict'], res['curr'], res['color']))
        bar.empty()
        for t, ver, curr, clr in results:
            st.markdown(f"<div class='fun-card' style='border-left:5px solid {clr}'><b>{t}</b>: {ver} @ ₹{curr:.2f}</div>", unsafe_allow_html=True)
