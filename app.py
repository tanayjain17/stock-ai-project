# -------------------- IMPORTS --------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import requests
from datetime import datetime
from io import StringIO

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

# -------------------- YFINANCE SAFE --------------------
@st.cache_data(ttl=300)
def yf_safe(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# -------------------- NSE BHAVCOPY --------------------
@st.cache_data(ttl=86400)
def download_nse_bhavcopy(date):
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    date_str = date_obj.strftime("%d%b%Y").upper()

    url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return None

    return pd.read_csv(StringIO(r.text))

def get_delivery_data(symbol, date):
    bhav = download_nse_bhavcopy(date)
    if bhav is None:
        return np.nan

    row = bhav[(bhav["SYMBOL"] == symbol) & (bhav["SERIES"] == "EQ")]
    if row.empty:
        return np.nan

    return (row["DELIV_QTY"].values[0] / row["TOTTRDQTY"].values[0]) * 100

# -------------------- FEATURES --------------------
def add_features(df):
    ma_periods = [10, 20, 50, 100, 200]

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
    df["VolAvg"] = df["Volume"].rolling(20).mean()

    above_ma = sum((df["Close"] > df[f"MA_{p}"]).astype(int) for p in ma_periods)
    df["Bullish_Score"] = (
        (df["RSI"] / 100) * 0.3 +
        (df["MACD"] > df["MACD_signal"]).astype(int) * 0.3 +
        (above_ma / len(ma_periods)) * 0.4
    )

    return df

def add_real_delivery(df, ticker):
    symbol = ticker.replace(".NS", "")
    df["Delivery_Pct"] = [
        get_delivery_data(symbol, d.strftime("%Y-%m-%d")) for d in df.index
    ]

    df["Delivery_5D"] = df["Delivery_Pct"].rolling(5).mean()
    df["Delivery_20D"] = df["Delivery_Pct"].rolling(20).mean()
    df["Delivery_Trend"] = (df["Delivery_5D"] > df["Delivery_20D"]).astype(int)

    return df

def add_support_resistance(df):
    high, low, close = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
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
        df = add_support_resistance(add_real_delivery(add_features(df), ticker))

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
        df = add_support_resistance(add_real_delivery(add_features(df), ticker))
        verdict, color, sl, tgt = ai_signal(df)
        c = df.iloc[-1]

        st.markdown(f"""
        ## <span style='color:{color}'>{verdict}</span>
        **Price:** ₹{c.Close:.2f}  
        **Bullish Score:** `{c.Bullish_Score:.2f}`  
        **RSI:** `{c.RSI:.2f}`  
        **Delivery %:** `{c.Delivery_Pct:.2f}%`

        **R1:** ₹{c.R1:.2f} | **R2:** ₹{c.R2:.2f} | **R3:** ₹{c.R3:.2f}  
        **S1:** ₹{c.S1:.2f} | **S2:** ₹{c.S2:.2f} | **S3:** ₹{c.S3:.2f}

        **Stoploss:** ₹{sl:.2f}  
        **Target:** ₹{tgt:.2f}
        """, unsafe_allow_html=True)

# ==================== PAGE 4 ====================
elif page == "⭐ Top 5 AI Picks":
    st.title("⭐ AI Top Picks (Live Market)")

    STOCKS = get_nifty50()

    min_score = st.slider("Minimum Bullish Score", 0.55, 0.80, 0.65, 0.01)

    results = []

    with st.spinner("AI scanning market..."):
        for s in STOCKS:
            df = yf_safe(s, "6mo", "1d")
            if df is None or len(df) < 120:
                continue

            df = add_support_resistance(add_real_delivery(add_features(df), s))
            verdict, _, _, _ = ai_signal(df)
            c = df.iloc[-1]

            if verdict in ["BUY 📈", "STRONG BUY 🚀"] and c.Delivery_Trend == 1:
                results.append({
                    "Stock": s,
                    "Price": round(c.Close, 2),
                    "RSI": round(c.RSI, 1),
                    "Bullish Score": round(c.Bullish_Score, 2),
                    "Delivery %": round(c.Delivery_Pct, 1)
                })

    if not results:
        st.warning("⚠️ No high-probability BUY setups today. Market is weak.")
    else:
        df_out = pd.DataFrame(results).sort_values(
            by="Bullish Score", ascending=False
        ).head(5)

        st.success("✅ AI-selected stocks for TODAY")
        st.dataframe(df_out, use_container_width=True)

# ==================== PAGE 5 ====================
elif page == "📰 Market News":
    st.title("📰 Market News")
    feed = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
    for e in feed.entries[:10]:
        st.write(f"• {e.title}")
