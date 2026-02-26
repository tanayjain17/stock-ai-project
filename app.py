# -------------------- IMPORTS --------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser

st.set_page_config("AI Trading Dashboard", layout="wide", page_icon="🚀")

# -------------------- SIDEBAR --------------------
PAGES = [
    "🏠 Market Dashboard",
    "📈 Chart & Technicals",
    "🤖 AI Prediction Center",
    "⭐ Top 5 AI Picks",
    "📰 Market News"
]
page = st.sidebar.radio("Navigation", PAGES)

# -------------------- YFINANCE SAFE WRAPPER --------------------
@st.cache_data(ttl=300)
def yf_safe(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# -------------------- TECHNICAL FEATURES --------------------
def add_features(df):
    ma_periods = [10, 20, 50, 100, 200]

    for p in ma_periods:
        df[f"MA_{p}"] = df["Close"].rolling(p).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ATR
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    # Volume
    df["VolAvg"] = df["Volume"].rolling(20).mean()

    # Delivery % (Estimated)
    df["Delivery_Pct"] = np.where(
        df["Volume"] > df["VolAvg"],
        (df["VolAvg"] / df["Volume"]) * 100,
        30
    )

    df["Delivery_5D"] = df["Delivery_Pct"].rolling(5).mean()
    df["Delivery_20D"] = df["Delivery_Pct"].rolling(20).mean()
    df["Delivery_Trend"] = (df["Delivery_5D"] > df["Delivery_20D"]).astype(int)

    above_ma = sum((df["Close"] > df[f"MA_{p}"]).astype(int) for p in ma_periods)

    df["Bullish_Score"] = (
        (df["RSI"] / 100) * 0.3 +
        (df["MACD"] > df["MACD_signal"]).astype(int) * 0.3 +
        (above_ma / len(ma_periods)) * 0.4
    )

    return df.dropna()

# -------------------- SUPPORT & RESISTANCE --------------------
def add_support_resistance(df):
    high = df["High"].shift(1)
    low = df["Low"].shift(1)
    close = df["Close"].shift(1)

    pivot = (high + low + close) / 3

    df["R1"] = 2 * pivot - low
    df["S1"] = 2 * pivot - high
    df["R2"] = pivot + (high - low)
    df["S2"] = pivot - (high - low)
    df["R3"] = high + 2 * (pivot - low)
    df["S3"] = low - 2 * (high - pivot)

    return df

# -------------------- AI SIGNAL --------------------
def ai_signal(df):
    c = df.iloc[-1]
    verdict, color = "HOLD ⚪", "gray"

    if c.Bullish_Score > 0.75 and c.Delivery_Trend == 1:
        verdict, color = "STRONG BUY 🚀", "#00d09c"
    elif c.Bullish_Score > 0.6:
        verdict, color = "BUY 📈", "#00d09c"
    elif c.RSI < 35:
        verdict, color = "DIP BUY 🟢", "#00d09c"
    elif c.Bullish_Score < 0.4:
        verdict, color = "SELL 🔻", "#ff4b4b"

    sl = c.Close - 1.5 * c.ATR
    tgt = c.Close + 2.5 * c.ATR
    return verdict, color, sl, tgt

# ==================== PAGE 1 ====================
if page == "🏠 Market Dashboard":
    st.title("📊 Market Dashboard")

    ASSETS = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "GOLD ETF": "GOLDBEES.NS",
        "SILVER ETF": "SILVERBEES.NS"
    }

    cols = st.columns(len(ASSETS))
    for (name, sym), col in zip(ASSETS.items(), cols):
        df = yf_safe(sym, "5d", "1d")
        if df is None:
            continue
        last, prev = df.Close.iloc[-1], df.Close.iloc[-2]
        col.metric(name, f"{last:,.2f}", f"{((last-prev)/prev)*100:+.2f}%")

# ==================== PAGE 2 ====================
elif page == "📈 Chart & Technicals":
    st.title("📈 Chart & Technical Analyzer")

    ticker = st.text_input("Ticker", "RELIANCE.NS")
    df = yf_safe(ticker, "6mo", "1d")

    if df is not None:
        df = add_support_resistance(add_features(df))

        fig = go.Figure(go.Candlestick(
            x=df.index, open=df.Open, high=df.High,
            low=df.Low, close=df.Close
        ))

        for lvl in ["R1","R2","R3","S1","S2","S3"]:
            fig.add_hline(y=df[lvl].iloc[-1], line_dash="dot", opacity=0.4)

        fig.add_scatter(x=df.index, y=df.MA_20, name="MA20")
        fig.add_scatter(x=df.index, y=df.MA_50, name="MA50")

        fig.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE 3 ====================
elif page == "🤖 AI Prediction Center":
    st.title("🤖 AI Prediction Center")

    ticker = st.text_input("Ticker", "RELIANCE.NS")
    df = yf_safe(ticker, "6mo", "1d")

    if df is not None:
        df = add_support_resistance(add_features(df))
        verdict, color, sl, tgt = ai_signal(df)
        c = df.iloc[-1]

        st.markdown(f"""
        ## <span style='color:{color}'>{verdict}</span>

        **Price:** ₹{c.Close:.2f}  
        **Bullish Score:** `{c.Bullish_Score:.2f}`  
        **RSI:** `{c.RSI:.2f}`  
        **Delivery %:** `{c.Delivery_Pct:.2f}%`

        ### 📌 Support & Resistance
        R1: ₹{c.R1:.2f} | R2: ₹{c.R2:.2f} | R3: ₹{c.R3:.2f}  
        S1: ₹{c.S1:.2f} | S2: ₹{c.S2:.2f} | S3: ₹{c.S3:.2f}

        **Stoploss:** ₹{sl:.2f}  
        **Target:** ₹{tgt:.2f}
        """, unsafe_allow_html=True)

# ==================== PAGE 4 ====================
elif page == "⭐ Top 5 AI Picks":
    st.title("⭐ Top 5 AI Picks")

    STOCKS = ["RELIANCE.NS","HDFCBANK.NS","INFY.NS","TCS.NS","ITC.NS"]

    if st.button("Run AI Scan"):
        for s in STOCKS:
            df = yf_safe(s, "6mo", "1d")
            if df is None:
                continue
            df = add_support_resistance(add_features(df))
            verdict, color, _, _ = ai_signal(df)
            st.markdown(f"<span style='color:{color}'><b>{s}</b> → {verdict}</span>",
                        unsafe_allow_html=True)

# ==================== PAGE 5 ====================
elif page == "📰 Market News":
    st.title("📰 Market News")
    feed = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
    for e in feed.entries[:10]:
        st.write(f"• {e.title}")
