import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import feedparser
from streamlit_autorefresh import st_autorefresh

# -------------------------------------------------
# PAGE CONFIG
st.set_page_config(page_title="Pro Market Dashboard", layout="wide", page_icon="📈")

# -------------------------------------------------
# AUTO REFRESH (30s)
st_autorefresh(interval=30000, key="refresh")

# -------------------------------------------------
# HELPERS

def smart_download(ticker, period):
    try:
        if ticker.startswith("^"):
            interval = "15m" if period in ["1d", "5d"] else "1d"
        else:
            if period == "1d":
                interval = "5m"
            elif period == "5d":
                interval = "15m"
            elif period == "1mo":
                interval = "60m"
            else:
                interval = "1d"

        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return None if df.empty else df
    except:
        return None


def get_live_price(ticker):
    try:
        return yf.Ticker(ticker).fast_info.last_price
    except:
        return None


def generate_ai_signal(df):
    curr = df['Close'].iloc[-1]
    sma = df['Close'].rolling(20).mean().iloc[-1]
    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]

    signal = "BUY" if curr > sma else "SELL"
    sl = curr - 1.5 * atr if signal == "BUY" else curr + 1.5 * atr
    tgt1 = curr + 2 * atr if signal == "BUY" else curr - 2 * atr
    tgt2 = curr + 4 * atr if signal == "BUY" else curr - 4 * atr

    return signal, sl, tgt1, tgt2


# -------------------------------------------------
# CONSTANTS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
SCANNER_POOL = ["RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS"]

# -------------------------------------------------
# DASHBOARD
st.title("📊 Market Dashboard")
cols = st.columns(3)

for (name, sym), col in zip(INDICES.items(), cols):
    df = smart_download(sym, "5d")
    if df is not None:
        live = get_live_price(sym)
        curr = live if live else df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
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

# -------------------------------------------------
# STOCK ANALYZER
st.markdown("---")
st.header("📈 Stock Analyzer")

sym = st.text_input("Enter NSE Stock (example: RELIANCE, INFY)", "RELIANCE")
ticker = f"{sym.upper()}.NS"

tf = st.radio("Timeframe", ["1d","5d","1mo","6mo","1y"], horizontal=True)
df = smart_download(ticker, tf)

if df is not None:
    live = get_live_price(ticker)
    curr = live if live else df['Close'].iloc[-1]

    # ---------- CHART ----------
    df_plot = df.copy().reset_index()

    if tf in ["1d", "5d"]:
        x_axis = list(range(len(df_plot)))  # continuous candles
        hover_time = df_plot.iloc[:, 0]
        show_ticks = False
    else:
        x_axis = df_plot.iloc[:, 0]
        hover_time = df_plot.iloc[:, 0]
        show_ticks = True

    fig = go.Figure(data=[go.Candlestick(
        x=x_axis,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        customdata=hover_time,
        hovertemplate=
        "Time: %{customdata}<br>" +
        "Open: %{open}<br>" +
        "High: %{high}<br>" +
        "Low: %{low}<br>" +
        "Close: %{close}<extra></extra>"
    )])

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False,
        xaxis=dict(showticklabels=show_ticks)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- AI TRADE PLAN ----------
    sig, sl, tgt1, tgt2 = generate_ai_signal(df)
    clr = "#00d09c" if sig == "BUY" else "#ff4b4b"

    st.markdown(f"""
    <div style="background:#161b22;border:1px solid {clr};padding:25px;border-radius:14px;margin-top:15px">
        <h2 style="color:{clr}">🤖 AI Trade Plan — {sig}</h2>
        <h1>Current Price: ₹{curr:.2f}</h1>
        <hr>
        <div style="display:flex;justify-content:space-between;font-size:20px">
            <div>🔵 ENTRY<br><b>₹{curr:.2f}</b></div>
            <div>🛑 STOP LOSS<br><b style="color:#ff4b4b">₹{sl:.2f}</b></div>
            <div>🎯 TARGET 1<br><b style="color:#00d09c">₹{tgt1:.2f}</b></div>
            <div>🚀 TARGET 2<br><b style="color:#00d09c">₹{tgt2:.2f}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("No data found. Check symbol.")

# -------------------------------------------------
# AI PICKS SCANNER
st.markdown("---")
st.header("⭐ Top 5 AI Picks")

if st.button("Run Scanner"):
    for t in SCANNER_POOL:
        df = smart_download(t, "6mo")
        if df is not None:
            sig, _, _, _ = generate_ai_signal(df)
            curr = df['Close'].iloc[-1]
            clr = "#00d09c" if sig=="BUY" else "#ff4b4b"
            st.markdown(
                f"<div style='padding:10px;border-left:5px solid {clr}'><b>{t}</b> → {sig} @ ₹{curr:.2f}</div>",
                unsafe_allow_html=True
            )

# -------------------------------------------------
# NEWS
st.markdown("---")
st.header("📰 Market News")

try:
    feed = feedparser.parse("https://www.moneycontrol.com/rss/marketreports.xml")
    for e in feed.entries[:5]:
        st.markdown(f"- [{e.title}]({e.link})")
except:
    st.write("News unavailable.")
