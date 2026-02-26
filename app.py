# ===================== IMPORTS =====================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config("AI Trading Dashboard", layout="wide", page_icon="🚀")

# ===================== SIDEBAR =====================
PAGES = [
    "🏠 Market Dashboard",
    "⚡ Intraday ML (XGBoost)",
    "📊 NSE Options Chain AI",
    "📈 AI Performance Dashboard"
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
    df = yf_safe("^NSEI", "3mo", "1d")
    if df is None or len(df) < 50:
        return False
    df["MA50"] = df["Close"].rolling(50).mean()
    return df.Close.iloc[-1] > df.MA50.iloc[-1]

# ===================== NSE OPTIONS CHAIN =====================
@st.cache_data(ttl=60)
def get_nse_options(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    }
    s = requests.Session()
    s.get("https://www.nseindia.com", headers=headers)
    r = s.get(url, headers=headers)
    if r.status_code != 200:
        return None

    rows = []
    for d in r.json()["records"]["data"]:
        if "CE" in d and "PE" in d:
            rows.append({
                "Strike": d["strikePrice"],
                "CE_OI": d["CE"]["openInterest"],
                "PE_OI": d["PE"]["openInterest"],
                "CE_IV": d["CE"]["impliedVolatility"],
                "PE_IV": d["PE"]["impliedVolatility"]
            })
    return pd.DataFrame(rows)

def options_ai_signal(df):
    pcr = round(df["PE_OI"].sum() / df["CE_OI"].sum(), 2)
    iv = round(df[["CE_IV","PE_IV"]].mean().mean(), 2)
    trend = is_market_bullish()

    if trend and pcr < 0.8:
        strat = "BUY CALL 📈"
    elif not trend and pcr > 1.2:
        strat = "BUY PUT 📉"
    elif iv > 18:
        strat = "SHORT STRADDLE 💰"
    else:
        strat = "NO TRADE ⚪"

    return pcr, iv, strat

# ===================== INTRADAY ML =====================
def add_features(df):
    df["Return"] = df["Close"].pct_change()
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["Target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
    return df.dropna()

def xgboost_signal(df):
    X = df[["Return","EMA9","EMA21","ATR"]]
    y = df["Target"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss"
    )
    model.fit(Xs, y)

    prob = model.predict_proba(scaler.transform([X.iloc[-1]]))[0][1] * 100

    signal = "BUY 🚀" if prob > 65 else "SELL 🔻" if prob < 35 else "WAIT ⚪"
    return signal, round(prob,1)

# ===================== RISK MANAGEMENT =====================
def atr_sl_target(df, rr=2):
    atr = df["ATR"].iloc[-1]
    entry = df["Close"].iloc[-1]
    return round(entry-atr,2), round(entry+(atr*rr),2)

def capital_allocation(prob, capital=100000):
    risk = min(max((prob-50)/50, 0.01), 0.03)
    return round(capital * risk, 0)

# ===================== JOURNAL =====================
def log_trade(stock, signal, entry, sl, tgt, prob):
    row = {
        "Date": datetime.now(),
        "Stock": stock,
        "Signal": signal,
        "Entry": entry,
        "SL": sl,
        "Target": tgt,
        "Prob": prob
    }
    try:
        df = pd.read_csv("trade_journal.csv")
        df = pd.concat([df, pd.DataFrame([row])])
    except:
        df = pd.DataFrame([row])
    df.to_csv("trade_journal.csv", index=False)

# ===================== PAGES =====================
if page == "🏠 Market Dashboard":
    st.title("📊 Market Dashboard")
    st.metric("Market Trend", "BULLISH" if is_market_bullish() else "BEARISH")

elif page == "⚡ Intraday ML (XGBoost)":
    st.title("⚡ Intraday AI Trading")

    ticker = st.text_input("Stock", "RELIANCE.NS")
    df = yf_safe(ticker, "5d", "5m")

    if df is not None:
        df = add_features(df)
        signal, prob = xgboost_signal(df)
        sl, tgt = atr_sl_target(df)
        capital = capital_allocation(prob)

        st.metric("Signal", signal)
        st.metric("Win Probability", f"{prob}%")
        st.metric("Capital Allocation", f"₹{capital}")
        st.metric("Stoploss", sl)
        st.metric("Target", tgt)

elif page == "📊 NSE Options Chain AI":
    st.title("📊 Options Strategy AI")
    symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY"])
    df = get_nse_options(symbol)

    if df is not None:
        pcr, iv, strat = options_ai_signal(df)
        st.metric("PCR", pcr)
        st.metric("IV", iv)
        st.success(strat)
        st.dataframe(df.sort_values("Strike"))

elif page == "📈 AI Performance Dashboard":
    st.title("📈 AI Accuracy Dashboard")
    try:
        df = pd.read_csv("trade_journal.csv")
        accuracy = (df["Target"] > df["Entry"]).mean() * 100
        st.metric("AI Accuracy", f"{round(accuracy,2)}%")
        st.dataframe(df.tail(20))
    except:
        st.warning("No trades logged yet.")
