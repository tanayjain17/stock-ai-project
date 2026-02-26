# ===================== IMPORTS =====================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import requests
from datetime import datetime
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config("AI Trading Dashboard", layout="wide", page_icon="🚀")

# ===================== SIDEBAR =====================
PAGES = [
    "🏠 Market Dashboard",
    "📈 Chart & Technicals",
    "🤖 AI Prediction Center",
    "⭐ AI Top Picks",
    "📰 Market News"
]
page = st.sidebar.radio("Navigation", PAGES)

# ===================== YFINANCE SAFE =====================
@st.cache_data(ttl=300)
def yf_safe(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ===================== MARKET TREND FILTER =====================
def is_market_bullish():
    df = yf_safe("^NSEI", "6mo", "1d")
    if df is None or len(df) < 60:
        return False

    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    last = df.iloc[-1]
    return last.Close > last.MA50 and last.RSI > 50

# ===================== SECTOR ROTATION AI =====================
SECTORS = {
    "IT": "ITBEES.NS",
    "BANK": "BANKBEES.NS",
    "PHARMA": "PHARMABEES.NS",
    "METAL": "METALBEES.NS",
    "FMCG": "FMCGBEES.NS"
}

SECTOR_STOCKS = {
    "IT": ["INFY.NS","TCS.NS","WIPRO.NS"],
    "BANK": ["HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS"],
    "PHARMA": ["SUNPHARMA.NS","CIPLA.NS","DRREDDY.NS"],
    "METAL": ["TATASTEEL.NS","JSWSTEEL.NS"],
    "FMCG": ["ITC.NS","HINDUNILVR.NS","DABUR.NS"]
}

def strong_sectors():
    strong = []
    for name, etf in SECTORS.items():
        df = yf_safe(etf, "3mo", "1d")
        if df is None or len(df) < 30:
            continue

        df["MA20"] = df["Close"].rolling(20).mean()
        if df.Close.iloc[-1] > df.MA20.iloc[-1]:
            strong.append(name)

    return strong

# ===================== FEATURES =====================
def add_features(df):
    ma_periods = [10, 20, 50]

    for p in ma_periods:
        df[f"MA_{p}"] = df["Close"].rolling(p).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    df["Bullish_Score"] = (
        (df["RSI"] / 100) * 0.4 +
        (df["MACD"] > df["MACD_signal"]).astype(int) * 0.3 +
        (df["Close"] > df["MA_20"]).astype(int) * 0.3
    )

    return df.dropna()

# ===================== ML PROBABILITY MODEL =====================
def ml_probability(df):
    df = df.copy()
    df["Future"] = df["Close"].shift(-5)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)
    df = df.dropna()

    X = df[["RSI","MACD","Bullish_Score"]]
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    latest = scaler.transform([X.iloc[-1]])
    prob = model.predict_proba(latest)[0][1]

    return prob * 100

# ===================== AI SIGNAL =====================
def ai_signal(df):
    c = df.iloc[-1]

    if c.Bullish_Score > 0.7:
        verdict = "BUY 📈"
    else:
        verdict = "HOLD ⚪"

    sl = c.Close - 1.5 * c.ATR
    tgt = c.Close + 2.5 * c.ATR
    return verdict, sl, tgt

# ===================== PAGE 4: AI TOP PICKS =====================
elif page == "⭐ AI Top Picks":
    st.title("⭐ AI Top Picks (Market + Sector + ML)")

    if not is_market_bullish():
        st.error("🚨 Market Trend is BEARISH — AI will not trade today")
        st.stop()

    strong_sec = strong_sectors()
    st.success(f"🔥 Strong Sectors: {', '.join(strong_sec)}")

    stocks = []
    for sec in strong_sec:
        stocks.extend(SECTOR_STOCKS[sec])

    results = []

    with st.spinner("AI scanning best stocks..."):
        for s in stocks:
            df = yf_safe(s, "6mo", "1d")
            if df is None or len(df) < 120:
                continue

            df = add_features(df)
            verdict, sl, tgt = ai_signal(df)

            if verdict == "BUY 📈":
                prob = ml_probability(df)
                results.append({
                    "Stock": s,
                    "Price": round(df.Close.iloc[-1],2),
                    "Bullish Score": round(df.Bullish_Score.iloc[-1],2),
                    "Win Probability %": round(prob,1),
                    "Stoploss": round(sl,2),
                    "Target": round(tgt,2)
                })

    if not results:
        st.warning("No high-quality AI setups today.")
    else:
        out = pd.DataFrame(results).sort_values(
            by="Win Probability %", ascending=False
        ).head(5)

        st.dataframe(out, use_container_width=True)

# ===================== NEWS =====================
elif page == "📰 Market News":
    st.title("📰 Market News")
    feed = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
    for e in feed.entries[:10]:
        st.write(f"• {e.title}")
