# -------------------- IMPORTS --------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser

st.set_page_config("AI Trading Dashboard", layout="wide", page_icon="🚀")

# -------------------- SIDEBAR NAV --------------------
PAGES = [
    "🏠 Market Dashboard",
    "📈 Chart & Technicals",
    "🤖 AI Prediction Center",
    "⭐ Top 5 AI Picks",
    "📰 Market News"
]
page = st.sidebar.radio("Navigation", PAGES)

# -------------------- SAFE YFINANCE --------------------
@st.cache_data(ttl=300)
def yf_safe(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

# -------------------- INDICATORS --------------------
def add_indicators(df):
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100/(1+rs))

    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['VolAvg'] = df['Volume'].rolling(20).mean()

    return df.dropna()

# -------------------- AI LOGIC --------------------
def ai_signal(df):
    c = df.iloc[-1]
    verdict = "HOLD"
    color = "gray"

    if c.Volume > 1.3*c.VolAvg and c.Close > c.SMA20:
        verdict, color = "MOMENTUM BUY 🚀", "#00d09c"
    elif c.RSI < 35:
        verdict, color = "DIP BUY 🟢", "#00d09c"
    elif c.Close > c.SMA50 and c.RSI > 45:
        verdict, color = "TREND BUY 📈", "#00d09c"
    elif c.Close < c.SMA50:
        verdict, color = "SELL 🔻", "#ff4b4b"

    sl = c.Close - 1.5*c.ATR
    tgt = c.Close + 2.5*c.ATR

    return verdict, color, sl, tgt

# -------------------- TIMEFRAME MAP --------------------
TF_MAP = {
    "1 Day": ("1d", "5m"),
    "3 Days": ("3d", "15m"),
    "1 Week": ("7d", "30m"),
    "1 Month": ("1mo", "60m"),
    "3 Months": ("3mo", "1d"),
    "1 Year": ("1y", "1d"),
    "5 Years": ("5y", "1wk"),
    "Max": ("max", "1wk"),
}

# -------------------- PAGE 1: DASHBOARD --------------------
if page == "🏠 Market Dashboard":
    st.title("📊 Market Dashboard")

    ASSETS = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "GOLD ETF": "GOLDBEES.NS",
        "SILVER ETF": "SILVERBEES.NS",
        "CRUDE OIL": "CL=F",
        "GOLD FUT": "GC=F",
        "SILVER FUT": "SI=F",
    }

    cols = st.columns(len(ASSETS))
    for (name, sym), col in zip(ASSETS.items(), cols):
        df = yf_safe(sym, "5d", "1d")
        if df is None:
            continue
        last = float(df['Close'].iloc[-1])
        prev = float(df['Close'].iloc[-2])
        pct = ((last-prev)/prev)*100
        col.metric(name, f"{last:,.2f}", f"{pct:+.2f}%")

# -------------------- PAGE 2: CHART --------------------
elif page == "📈 Chart & Technicals":
    st.title("📈 Chart & Technical Analyzer")

    ticker = st.text_input("Ticker", "RELIANCE.NS")
    tf = st.selectbox("Timeframe", list(TF_MAP.keys()))
    period, interval = TF_MAP[tf]

    df = yf_safe(ticker, period, interval)

    if df is not None:
        df = add_indicators(df)

        fig = go.Figure(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ))
        fig.add_scatter(x=df.index, y=df['SMA20'], name="SMA20")
        fig.add_scatter(x=df.index, y=df['SMA50'], name="SMA50")
        fig.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- PAGE 3: AI CENTER --------------------
elif page == "🤖 AI Prediction Center":
    st.title("🤖 AI Prediction Center")

    ticker = st.text_input("Ticker for AI Analysis", "RELIANCE.NS")
    df = yf_safe(ticker, "6mo", "1d")

    if df is not None:
        df = add_indicators(df)
        verdict, color, sl, tgt = ai_signal(df)

        st.markdown(f"""
        ## <span style='color:{color}'>{verdict}</span>
        **Current Price:** ₹{df.Close.iloc[-1]:.2f}  
        **Stoploss:** ₹{sl:.2f}  
        **Target:** ₹{tgt:.2f}
        """, unsafe_allow_html=True)

# -------------------- PAGE 4: SCANNER --------------------
elif page == "⭐ Top 5 AI Picks":
    st.title("⭐ Top 5 AI Picks Scanner")

    SCAN = ["RELIANCE.NS","HDFCBANK.NS","INFY.NS","TCS.NS","ITC.NS"]

    if st.button("Run AI Scan"):
        for s in SCAN:
            df = yf_safe(s, "6mo", "1d")
            if df is None:
                continue
            df = add_indicators(df)
            verdict, color, _, _ = ai_signal(df)
            st.markdown(f"<span style='color:{color}'><b>{s}</b> → {verdict}</span>", unsafe_allow_html=True)

# -------------------- PAGE 5: NEWS --------------------
elif page == "📰 Market News":
    st.title("📰 Market News")

    feed = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
    for e in feed.entries[:10]:
        st.write(f"• {e.title}")
