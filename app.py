# ========================= app.py =========================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

st.set_page_config(layout="wide", page_title="AI Trading Dashboard")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model, features = joblib.load("model.pkl")
    return model, features

model, FEATURES = load_model()

# ---------- Feature Engineering ----------
def make_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['EMA20'] = EMAIndicator(df['Close'], 20).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close']).rsi()

    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    bb = BollingerBands(df['Close'])
    df['BB_H'] = bb.bollinger_hband()
    df['BB_L'] = bb.bollinger_lband()
    df['BB_W'] = df['BB_H'] - df['BB_L']

    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    df.dropna(inplace=True)
    return df

# ---------- UI ----------
st.title("📈 AI Trading Dashboard")

symbol = st.text_input("Symbol", "RELIANCE.NS")

df = yf.download(symbol, period="6mo", interval="1d", auto_adjust=True)

df = make_features(df)

latest = df.iloc[-1:][FEATURES]
proba = model.predict_proba(latest)[0][1]

price = df['Close'].iloc[-1]
atr = df['ATR'].iloc[-1]

# ---------- Trade Signal ----------
if proba > 0.6:
    signal = "BUY 🚀"
    sl = price - 1.5 * atr
    target = price + 3 * atr
elif proba < 0.4:
    signal = "SELL 🔻"
    sl = price + 1.5 * atr
    target = price - 3 * atr
else:
    signal = "HOLD ⏸️"
    sl, target = None, None

# ---------- Metrics ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"₹{price:.2f}")
c2.metric("AI Confidence", f"{proba*100:.1f}%")
c3.metric("Signal", signal)

if sl:
    c4.metric("Stoploss", f"₹{sl:.2f}")

# ---------- Chart ----------
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

if target:
    st.success(f"🎯 Target: ₹{target:.2f}")
