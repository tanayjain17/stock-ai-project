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

# ===================== MARKET TREND =====================
def is_market_bullish():
    df = yf_safe("^NSEI", "3mo", "1d")
    if df is None or len(df) < 50:
        return False
    df["MA50"] = df["Close"].rolling(50).mean()
    return df.Close.iloc[-1] > df.MA50.iloc[-1]

# =====================================================
# NSE OPTIONS CHAIN (LIVE OI)
# =====================================================
@st.cache_data(ttl=60)
def get_nse_options(symbol):
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

    data = r.json()["records"]["data"]
    rows = []

    for d in data:
        if "CE" in d and "PE" in d:
            rows.append({
                "Strike": d["strikePrice"],
                "CE_OI": d["CE"]["openInterest"],
                "PE_OI": d["PE"]["openInterest"],
                "CE_IV": d["CE"]["impliedVolatility"],
                "PE_IV": d["PE"]["impliedVolatility"],
            })

    return pd.DataFrame(rows)

def options_ai_signal(df):
    ce_oi = df["CE_OI"].sum()
    pe_oi = df["PE_OI"].sum()
    pcr = round(pe_oi / ce_oi, 2)
    iv = df[["CE_IV","PE_IV"]].mean().mean()

    if is_market_bullish():
        if pcr < 0.8:
            strategy = "BUY CALL 📈"
        elif pcr > 1.2:
            strategy = "BUY PUT 📉"
        else:
            strategy = "NO TRADE ⚪"
    else:
        strategy = "SHORT STRADDLE 💰" if iv > 18 else "LONG STRADDLE ⚡"

    return pcr, iv, strategy

# =====================================================
# INTRADAY XGBOOST ML
# =====================================================
def add_intraday_features(df):
    df["Return"] = df["Close"].pct_change()
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (
        df["Close"].diff().clip(lower=0).rolling(14).mean() /
        df["Close"].diff().clip(upper=0).abs().rolling(14).mean()
    )))
    df["Target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
    return df.dropna()

def xgboost_intraday_signal(df):
    X = df[["Return","EMA9","EMA21","RSI"]]
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

    if prob > 65:
        return "SCALP BUY 🚀", round(prob,1)
    elif prob < 35:
        return "SCALP SELL 🔻", round(100-prob,1)
    return "WAIT ⚪", round(prob,1)

def atr_sl_target(df, rr=2):
    atr = df["ATR"].iloc[-1]
    entry = df["Close"].iloc[-1]
    return round(entry - atr,2), round(entry + atr*rr,2)

def log_trade(stock, signal, entry, sl, tgt, prob):
    row = {
        "Date": datetime.now(),
        "Stock": stock,
        "Signal": signal,
        "Entry": entry,
        "Stoploss": sl,
        "Target": tgt,
        "Probability": prob
    }

    try:
        df = pd.read_csv("trade_journal.csv")
        df = pd.concat([df, pd.DataFrame([row])])
    except:
        df = pd.DataFrame([row])

    df.to_csv("trade_journal.csv", index=False)

# ===================== PAGE 1 =====================
if page == "🏠 Market Dashboard":
    st.title("📊 Market Dashboard")
    st.metric("Market Trend", "BULLISH" if is_market_bullish() else "BEARISH")

# ===================== PAGE 2 =====================
elif page == "⚡ Intraday ML (XGBoost)":
    st.title("⚡ Intraday XGBoost Scalping (5-Min)")

    ticker = st.text_input("Stock", "RELIANCE.NS")
    df = yf_safe(ticker, "5d", "5m")

    if df is not None and len(df) > 50:
        df = add_intraday_features(df)
        signal, prob = xgboost_intraday_signal(df)
        sl, tgt = atr_sl_target(df)

        st.metric("Signal", signal)
        st.metric("Win Probability", f"{prob}%")
        st.metric("Stoploss", sl)
        st.metric("Target", tgt)

        if st.button("Log Trade"):
            log_trade(ticker, signal, df.Close.iloc[-1], sl, tgt, prob)
            st.success("Trade logged successfully")

# ===================== PAGE 3 =====================
elif page == "📊 NSE Options Chain AI":
    st.title("📊 NSE Options Chain AI")

    symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY"])
    df = get_nse_options(symbol)

    if df is None:
        st.error("NSE blocked. Try again later.")
    else:
        pcr, iv, strategy = options_ai_signal(df)
        st.metric("PCR", pcr)
        st.metric("Avg IV", round(iv,1))
        st.success(strategy)
        st.dataframe(df.sort_values("Strike"))
