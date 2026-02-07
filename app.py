import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Pro Trading Dashboard", layout="wide", page_icon="📈")
st_autorefresh(interval=30000, key="refresh")

# -------------------- ASSETS --------------------
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}

ETFS = {
    "Nifty ETF": "NIFTYBEES.NS",
    "Bank ETF": "BANKBEES.NS",
    "Gold ETF": "GOLDBEES.NS",
    "Silver ETF": "SILVERBEES.NS",
    "Nasdaq ETF": "MON100.NS"
}

COMMODITIES = {
    "Gold (Global)": "GC=F",
    "Silver (Global)": "SI=F",
    "Crude Oil": "CL=F"
}

SCANNER_POOL = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS",
    "INFY.NS","TCS.NS","ITC.NS","LT.NS"
]

# -------------------- HELPERS --------------------
def smart_download(ticker, period):
    interval_map = {
        "1d": "5m",
        "5d": "15m",
        "1mo": "60m",
        "6mo": "1d",
        "1y": "1d"
    }
    interval = interval_map.get(period, "1d")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return None if df.empty else df


def add_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # ATR
    hl = df["High"] - df["Low"]
    hc = abs(df["High"] - df["Close"].shift())
    lc = abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()


def ai_signal(df):
    curr = df["Close"].iloc[-1]
    ema = df["EMA20"].iloc[-1]
    atr = df["ATR"].iloc[-1]

    signal = "BUY" if curr > ema else "SELL"
    sl = curr - 1.5 * atr if signal == "BUY" else curr + 1.5 * atr
    t1 = curr + 2 * atr if signal == "BUY" else curr - 2 * atr
    t2 = curr + 4 * atr if signal == "BUY" else curr - 4 * atr
    return signal, sl, t1, t2


def plot_chart(df, title):
    fig = go.Figure()

    # remove gaps
    fig.update_xaxes(type='category')

    fig.add_trace(go.Candlestick(
        x=df.index.strftime('%Y-%m-%d %H:%M'),
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(x=df.index.strftime('%Y-%m-%d %H:%M'),
                             y=df["EMA20"], name="EMA20"))

    fig.add_trace(go.Scatter(x=df.index.strftime('%Y-%m-%d %H:%M'),
                             y=df["SMA50"], name="SMA50"))

    fig.update_layout(template="plotly_dark",
                      height=500,
                      xaxis_rangeslider_visible=False,
                      title=title)

    st.plotly_chart(fig, use_container_width=True)


# -------------------- DASHBOARD --------------------
st.title("🌏 Market Dashboard")
cols = st.columns(3)

for (name, sym), col in zip(INDICES.items(), cols):
    df = smart_download(sym, "5d")
    if df is not None:
        curr = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2]
        chg = curr - prev
        pct = chg / prev * 100
        clr = "#00d09c" if chg >= 0 else "#ff4b4b"

        with col:
            st.markdown(f"""
            <div style="background:#161b22;padding:18px;border-radius:14px;border-top:4px solid {clr}">
                <div style="color:#aaa">{name}</div>
                <div style="font-size:26px;font-weight:bold">₹{curr:,.2f}</div>
                <div style="color:{clr};font-weight:bold">{chg:+.2f} ({pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

# -------------------- ANALYZER --------------------
st.markdown("---")
st.header("📈 Stock / ETF / Commodity Analyzer")

asset = st.selectbox("Choose Asset",
                     list(ETFS.keys()) +
                     list(COMMODITIES.keys()))

ticker = ETFS.get(asset) or COMMODITIES.get(asset)

tf = st.radio("Timeframe", ["1d", "5d", "1mo", "6mo", "1y"], horizontal=True)
df = smart_download(ticker, tf)

if df is not None:
    df = add_indicators(df)
    plot_chart(df, asset)

    sig, sl, t1, t2 = ai_signal(df)
    curr = df["Close"].iloc[-1]
    clr = "#00d09c" if sig == "BUY" else "#ff4b4b"

    st.markdown(f"""
    <div style="background:#161b22;border:1px solid {clr};
    padding:20px;border-radius:14px;margin-top:10px">
        <h2 style="color:{clr}">AI Trade Plan — {sig}</h2>
        <b>Entry:</b> ₹{curr:.2f} |
        <b>SL:</b> ₹{sl:.2f} |
        <b>T1:</b> ₹{t1:.2f} |
        <b>T2:</b> ₹{t2:.2f}
    </div>
    """, unsafe_allow_html=True)

# -------------------- TOP 5 SCANNER --------------------
st.markdown("---")
st.header("⭐ Top AI Picks")

if st.button("Run Scanner"):
    picks = []
    for t in SCANNER_POOL:
        df = smart_download(t, "6mo")
        if df is not None:
            df = add_indicators(df)
            sig, _, _, _ = ai_signal(df)
            if sig == "BUY":
                picks.append(t)

    for p in picks[:5]:
        st.write("✅", p)

# -------------------- NEWS --------------------
st.markdown("---")
st.header("📰 Market News")

feed = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
for e in feed.entries[:5]:
    st.markdown(f"- [{e.title}]({e.link})")
