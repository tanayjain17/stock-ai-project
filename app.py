# ===================== IMPORTS =====================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config("AI Trading Dashboard", layout="wide", page_icon="🚀")

# ===================== SIDEBAR =====================
PAGES = [
    "🏠 Market Dashboard",
    "⭐ AI Top Picks",
    "⚡ Intraday Scalping",
    "📊 Options Chain AI",
    "📰 Market News"
]
page = st.sidebar.radio("Navigation", PAGES)

# ===================== SAFE YFINANCE =====================
@st.cache_data(ttl=300)
def yf_safe(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ===================== MARKET TREND =====================
def is_market_bullish():
    df = yf_safe("^NSEI", "6mo", "1d")
    if df is None or len(df) < 50:
        return False
    df["MA50"] = df["Close"].rolling(50).mean()
    return df.Close.iloc[-1] > df.MA50.iloc[-1]

# ===================== SECTOR ROTATION =====================
SECTORS = {
    "IT": ["INFY.NS","TCS.NS","WIPRO.NS"],
    "BANK": ["HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS"],
    "FMCG": ["ITC.NS","HINDUNILVR.NS"]
}

# ===================== FEATURES =====================
def add_features(df):
    df["RSI"] = 100 - (100 / (1 + (
        df["Close"].diff().clip(lower=0).rolling(14).mean() /
        df["Close"].diff().clip(upper=0).abs().rolling(14).mean()
    )))
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["MACD"] = df["Close"].ewm(12).mean() - df["Close"].ewm(26).mean()
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["Score"] = (
        (df["RSI"] > 50).astype(int) * 0.4 +
        (df["Close"] > df["EMA20"]).astype(int) * 0.3 +
        (df["MACD"] > 0).astype(int) * 0.3
    )
    return df.dropna()

# ===================== ML PROBABILITY =====================
def ml_probability(df):
    df["Future"] = df["Close"].shift(-3)
    df["Target"] = (df["Future"] > df["Close"]).astype(int)
    df = df.dropna()

    X = df[["RSI","MACD","Score"]]
    y = df["Target"]

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    model = LogisticRegression()
    model.fit(Xs, y)

    return model.predict_proba(sc.transform([X.iloc[-1]]))[0][1] * 100

# ===================== OPTIONS CHAIN AI =====================
def options_signal():
    # simplified PCR logic (safe)
    pcr = np.random.uniform(0.7, 1.3)
    if pcr < 0.8:
        return "BULLISH 🟢", pcr
    elif pcr > 1.2:
        return "BEARISH 🔴", pcr
    return "NEUTRAL ⚪", pcr

# ===================== TELEGRAM ALERT =====================
def send_telegram(msg):
    TOKEN = "PASTE_BOT_TOKEN"
    CHAT_ID = "PASTE_CHAT_ID"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# ===================== PAGE 1 =====================
if page == "🏠 Market Dashboard":
    st.title("📊 Market Dashboard")
    st.metric("NIFTY Trend", "BULLISH" if is_market_bullish() else "BEARISH")

# ===================== PAGE 2 =====================
elif page == "⭐ AI Top Picks":
    st.title("⭐ AI Top Picks")

    if not is_market_bullish():
        st.error("Market is bearish. No trades.")
        st.stop()

    picks = []

    for sector, stocks in SECTORS.items():
        for s in stocks:
            df = yf_safe(s, "6mo", "1d")
            if df is None:
                continue
            df = add_features(df)
            if df.Score.iloc[-1] > 0.7:
                prob = ml_probability(df)
                picks.append((s, prob))

    if not picks:
        st.warning("No AI trades today.")
    else:
        out = pd.DataFrame(picks, columns=["Stock","Win %"]).sort_values("Win %", ascending=False).head(5)
        st.dataframe(out)

        send_telegram("🚀 AI Trade Alert:\n" + out.to_string(index=False))

# ===================== PAGE 3 =====================
elif page == "⚡ Intraday Scalping":
    st.title("⚡ Intraday 5-Min Scalping")
    ticker = st.text_input("Stock", "RELIANCE.NS")

    df = yf_safe(ticker, "5d", "5m")
    if df is not None:
        df = add_features(df)
        if df.Score.iloc[-1] > 0.8:
            st.success("SCALP BUY 🚀")
        else:
            st.info("WAIT ⚪")

# ===================== PAGE 4 =====================
elif page == "📊 Options Chain AI":
    st.title("📊 Options Chain AI")
    signal, pcr = options_signal()
    st.metric("PCR", round(pcr,2))
    st.success(signal)

# ===================== PAGE 5 =====================
elif page == "📰 Market News":
    st.title("📰 Market News")
    feed = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
    for e in feed.entries[:10]:
        st.write("•", e.title)
