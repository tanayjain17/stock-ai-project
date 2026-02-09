# ---------- IMPORTS ----------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
from datetime import datetime

st.set_page_config("Pro Market Dashboard", layout="wide", page_icon="🚀")

# ---------- SAFE YF ----------
@st.cache_data(ttl=300)
def yf_safe(ticker, period="1y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return None

    # FIX FOR CONTINUOUS CHART
    df = df.asfreq("D")
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].ffill()
    return df

# ---------- INDICATORS ----------
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

# ---------- QUANT LOGIC ----------
def quant_signal(df):
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

# ---------- CANDLE CHART ----------
def candle_chart(df):
    fig = go.Figure(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))
    fig.add_scatter(x=df.index, y=df['SMA20'], name="SMA20")
    fig.add_scatter(x=df.index, y=df['SMA50'], name="SMA50")
    fig.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig, use_container_width=True)

# ---------- ASSETS ----------
ASSETS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "GOLD ETF": "GOLDBEES.NS",
    "SILVER ETF": "SILVERBEES.NS",
    "NIFTY ETF": "NIFTYBEES.NS",
    "BANK ETF": "BANKBEES.NS",
    "CRUDE OIL": "CL=F",
    "GOLD FUT": "GC=F",
    "SILVER FUT": "SI=F",
}

# ---------- DASHBOARD ----------
st.title("🚀 Pro Market Dashboard")

cols = st.columns(len(ASSETS))
for (name, sym), col in zip(ASSETS.items(), cols):
    df = yf_safe(sym, "5d")
    if df is None: 
        continue
    last = df.Close.iloc[-1]
    prev = df.Close.iloc[-2]
    chg = last - prev
    pct = (chg/prev)*100
    color = "#00d09c" if chg>=0 else "#ff4b4b"

    col.metric(name, f"₹{last:,.2f}", f"{pct:+.2f}%")

st.divider()

# ---------- ANALYZER ----------
st.subheader("📈 Stock / ETF / Commodity Analyzer")
ticker = st.text_input("Enter Ticker (e.g. RELIANCE.NS, GOLDBEES.NS, GC=F)", "RELIANCE.NS")

df = yf_safe(ticker)
if df is not None:
    df = add_indicators(df)
    verdict, color, sl, tgt = quant_signal(df)

    st.markdown(f"""
    ### <span style='color:{color}'>{verdict}</span> @ ₹{df.Close.iloc[-1]:.2f}
    **Stoploss:** ₹{sl:.2f} | **Target:** ₹{tgt:.2f}
    """, unsafe_allow_html=True)

    candle_chart(df)

else:
    st.error("Ticker not found.")

st.divider()

# ---------- SCANNER ----------
st.subheader("⭐ Top Momentum Scanner")

SCAN_LIST = ["RELIANCE.NS","HDFCBANK.NS","INFY.NS","TCS.NS","ITC.NS","GOLDBEES.NS","SILVERBEES.NS"]

if st.button("Run Scan"):
    for s in SCAN_LIST:
        df = yf_safe(s, "6mo")
        if df is None: 
            continue
        df = add_indicators(df)
        verdict, color, _, _ = quant_signal(df)
        st.markdown(f"<span style='color:{color}'><b>{s}</b> → {verdict}</span>", unsafe_allow_html=True)

# ---------- NEWS ----------
st.subheader("📰 Market News")
feed = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
for e in feed.entries[:5]:
    st.write(f"• {e.title}")
