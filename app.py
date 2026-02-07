# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import gdown
import os
from datetime import datetime, time as dt_time
import pytz
import feedparser
from textblob import TextBlob
import urllib.parse
import time
import tensorflow as tf
from streamlit_autorefresh import st_autorefresh

# --------------------------
# 0. SEED
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------
# 1. AUTO REFRESH
st_autorefresh(interval=15000, key="dashboard_refresh")  # every 15 seconds

# --------------------------
# 2. PAGE STYLING
st.set_page_config(page_title="Market Pulse AI", layout="wide", page_icon="⚡")
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
.metric-card { background-color: #1e2330; border:1px solid #2a2f3d; border-radius:8px; padding:15px; text-align:center; }
.status-badge { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight:bold; text-transform:uppercase; letter-spacing:1px;}
.status-live { background-color:#00e676; color:black; }
.status-closed { background-color:#ff1744; color:white; }
.trade-plan { border:1px solid #4caf50; background-color:#1b2e1b; padding:15px; border-radius:10px; margin-top:10px;}
.news-card { background-color: #151922; padding:15px; border-radius:8px; margin-bottom:12px; border-left:4px solid #2962ff; transition: transform 0.2s;}
.news-card:hover { transform:translateX(5px); background-color:#1a1f2b; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 3. DATABASES
INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK"
}

NIFTY_100_TICKERS = {
    "Reliance Industries":"RELIANCE.NS","TCS":"TCS.NS","HDFC Bank":"HDFCBANK.NS",
    "ICICI Bank":"ICICIBANK.NS","Infosys":"INFY.NS","SBI":"SBIN.NS",
    "Bharti Airtel":"BHARTIARTL.NS","ITC":"ITC.NS","L&T":"LT.NS",
    "Kotak Bank":"KOTAKBANK.NS","Axis Bank":"AXISBANK.NS","Tata Motors":"TATAMOTORS.NS",
    "Maruti Suzuki":"MARUTI.NS","Bajaj Finance":"BAJFINANCE.NS","Sun Pharma":"SUNPHARMA.NS",
    "Titan":"TITAN.NS","NTPC":"NTPC.NS","Power Grid":"POWERGRID.NS","ONGC":"ONGC.NS",
    "Zomato":"ZOMATO.NS","Adani Ent":"ADANIENT.NS","DLF":"DLF.NS","HAL":"HAL.NS",
    "Tata Steel":"TATASTEEL.NS","Hindalco":"HINDALCO.NS","Jio Financial":"JIOFIN.NS"
}

ETFS_MFS = {
    "Nifty 50 ETF":"NIFTYBEES.NS",
    "Bank Nifty ETF":"BANKBEES.NS",
    "IT Tech ETF":"ITBEES.NS",
    "Pharma ETF":"PHARMABEES.NS",
    "Gold ETF":"GOLDBEES.NS",
    "Silver ETF":"SILVERBEES.NS",
    "US Nasdaq 100":"MON100.NS",
    "CPSE ETF":"CPSEETF.NS"
}

COMMODITIES_GLOBAL = {
    "Gold":"GC=F","Silver":"SI=F","Crude Oil":"CL=F","Natural Gas":"NG=F","Copper":"HG=F"
}

# --------------------------
# 4. UTILITIES
def is_market_open(ticker):
    utc_now = datetime.now(pytz.utc)
    ist_now = utc_now.astimezone(pytz.timezone('Asia/Kolkata'))
    if ist_now.weekday()>=5: return False, "Weekend"
    if dt_time(9,15)<=ist_now.time()<=dt_time(15,30): return True, "Open"
    return False, "Closed"

def get_currency(ticker):
    return "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"

def get_live_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.fast_info.last_price
        prev = stock.fast_info.previous_close
        if price is None or prev is None:
            return 0.0,0.0,0.0
        return price, price-prev, (price-prev)/prev*100
    except:
        return 0.0,0.0,0.0

def get_news(query, count=5):
    try:
        url=f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed=feedparser.parse(url)
        items=[]
        for e in feed.entries[:count]:
            blob = TextBlob(e.title)
            sent = "🟢 Bullish" if blob.sentiment.polarity>0.05 else "🔴 Bearish" if blob.sentiment.polarity<-0.05 else "⚪ Neutral"
            ts = time.mktime(e.published_parsed) if 'published_parsed' in e else 0
            items.append({'title':e.title,'link':e.link,'source':e.get('source',{}).get('title','News'),'date':e.get('published','')[:16],'ts':ts,'sent':sent})
        return sorted(items,key=lambda x:x['ts'],reverse=True)
    except: return []

def add_indicators(df):
    if df.empty: return df
    df['SMA_50']=df['Close'].rolling(50,min_periods=1).mean()
    df['EMA_20']=df['Close'].ewm(span=20,min_periods=1).mean()
    high_low = df['High']-df['Low']
    high_close = np.abs(df['High']-df['Close'].shift())
    low_close = np.abs(df['Low']-df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14,min_periods=1).mean()
    delta=df['Close'].diff()
    gain = delta.where(delta>0,0).rolling(14,min_periods=1).mean()
    loss = -delta.where(delta<0,0).rolling(14,min_periods=1).mean()
    rs = gain/loss.replace(0,np.nan)
    df['RSI'] = 100 - (100/(1+rs))
    ema12 = df['Close'].ewm(span=12,min_periods=1).mean()
    ema26 = df['Close'].ewm(span=26,min_periods=1).mean()
    df['MACD'] = ema12-ema26
    df['Signal'] = df['MACD'].ewm(span=9,min_periods=1).mean()
    return df.fillna(0)

# --------------------------
# 5. LOAD PRETRAINED MODEL
@st.cache_resource(show_spinner=True)
def load_pretrained_model():
    if not os.path.exists("stock_model.h5"):
        MODEL_ID = "YOUR_MODEL_FILE_ID"
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", "stock_model.h5", quiet=False)
    if not os.path.exists("stock_scaler.gz"):
        SCALER_ID = "YOUR_SCALER_FILE_ID"
        gdown.download(f"https://drive.google.com/uc?id={SCALER_ID}", "stock_scaler.gz", quiet=False)
    model = load_model("stock_model.h5")
    scaler = joblib.load("stock_scaler.gz")
    return model, scaler

pretrained_model, pretrained_scaler = load_pretrained_model()

# --------------------------
# 6. TRAIN AI FUNCTION
def train_ai(df):
    df_ai = df[['Close','RSI','SMA_50','EMA_20']].copy()
    df_ai = df_ai.fillna(0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_ai)
    if len(scaled)<60: return None,None
    X,y=[],[]
    for i in range(60,len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i,0])
    X,y = np.array(X), np.array(y)
    model=Sequential([
        LSTM(50,return_sequences=True,input_shape=(X.shape[1],4)),
        Dropout(0.2),
        LSTM(50,return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X,y,epochs=3,batch_size=16,verbose=0)
    last60 = scaled[-60:].reshape(1,60,4)
    pred_scaled = model.predict(last60, verbose=0)
    dummy = np.zeros((1,4))
    dummy[0,0] = pred_scaled[0,0]
    pred_price = scaler.inverse_transform(dummy)[0,0]
    return pred_price, df['ATR'].iloc[-1]

# --------------------------
# 7. CACHE AI PREDICTION
@st.cache_data(show_spinner=False)
def compute_ai_prediction(df):
    pred_price, atr = train_ai(df)
    return pred_price, atr

# --------------------------
# 8. SIDEBAR NAVIGATION
st.sidebar.title("⚡ Market Pulse AI")
nav_options=["🏠 Market Dashboard","📈 Stock Analyzer","🏦 ETFs & Mutual Funds","🛢️ Global Commodities","⭐ Top 5 AI Picks"]
view=st.sidebar.radio("Go to:",nav_options)
selected_ticker = "RELIANCE.NS"

# --------------------------
# 9. SELECT TICKER
if view=="📈 Stock Analyzer":
    t_name = st.sidebar.selectbox("Nifty 100 List",list(NIFTY_100_TICKERS.keys()))
    selected_ticker = NIFTY_100_TICKERS[t_name]
    custom = st.sidebar.text_input("Or Search Stock (e.g. ZOMATO)")
    if custom: selected_ticker=f"{custom.upper()}.NS"
elif view=="🏦 ETFs & Mutual Funds":
    t_name = st.sidebar.selectbox("Popular ETFs",list(ETFS_MFS.keys()))
    selected_ticker = ETFS_MFS[t_name]
    custom = st.sidebar.text_input("Or Search ETF")
    if custom: selected_ticker=f"{custom.upper()}.NS"
elif view=="🛢️ Global Commodities":
    t_name = st.sidebar.selectbox("Global Assets",list(COMMODITIES_GLOBAL.keys()))
    selected_ticker = COMMODITIES_GLOBAL[t_name]

# --------------------------
# 10. LIVE HEADER
is_open,_ = is_market_open(selected_ticker)
curr_sym = get_currency(selected_ticker)
lp,lc,lpct = get_live_data(selected_ticker)
color="#00e676" if lc>=0 else "#ff1744"
badge_html = f'<span class="status-badge status-live">🟢 LIVE</span>' if is_open else f'<span class="status-badge status-closed">🔴 CLOSED</span>'

st.markdown(f"""
<div style="background:#1e2330; padding:20px; border-radius:12px; margin-bottom:10px; border-left:6px solid {color};">
    <div style="display:flex; justify-content:space-between;">
        <div><div style="color:#aaa; font-size:12px;">MARKET STATUS &nbsp; {badge_html}</div>
        <div style="font-size:42px; font-weight:bold; color:white;">{curr_sym}{lp:,.2f}</div></div>
        <div style="text-align:right;">
            <div style="font-size:24px; font-weight:bold; color:{color};">{lc:+.2f}</div>
            <div style="background:{color}20; color:{color}; padding:4px 10px; border-radius:8px; font-weight:bold; display:inline-block;">{lpct:+.2f}%</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------
# 11. HISTORICAL CHART + INDICATORS
df_hist = yf.download(selected_ticker, period="1y", interval="1d")
df_hist = add_indicators(df_hist)

fig = go.Figure()
fig.add_trace(go.Candlestick(x=df_hist.index, open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name="Price"))
fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['SMA_50'], line=dict(color='green',width=2), name="SMA 50"))
fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['EMA_20'], line=dict(color='orange',width=2), name="EMA 20"))
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# 12. AI PREDICTION
st.subheader("🤖 AI Prediction")
try:
    pred_price, atr = compute_ai_prediction(df_hist)
    if pred_price:
        curr,_,_=get_live_data(selected_ticker)
        diff = pred_price-curr
        sig = "BUY 🚀" if diff>0 else "SELL 🔻"
        st.success(f"AI Target: {curr_sym}{pred_price:.2f}")
        st.markdown(f"Signal: **{sig}** (Potential: {diff:+.2f})")
        with st.expander("🔐 AI Trade Plan (Entry/SL/Targets)"):
            sl=curr-1.5*atr if diff>0 else curr+1.5*atr
            t1=curr+1*atr if diff>0 else curr-1*atr
            t2=curr+2*atr if diff>0 else curr-2*atr
            st.markdown(f"""
            <div class="trade-plan">
            <h4 style="color:#4caf50">AI Trade Setup</h4>
            <p>Entry: {curr:.2f}</p>
            <p>Stop Loss: {sl:.2f}</p>
            <p>Target 1: {t1:.2f}</p>
            <p>Target 2: {t2:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
except:
    st.warning("Prediction unavailable")

# --------------------------
# 13. MARKET DASHBOARD
if view=="🏠 Market Dashboard":
    st.title("🌏 Market Dashboard")
    c1,c2,c3 = st.columns(3)
    for (name,symbol),col in zip(INDICES.items(), [c1,c2,c3]):
        is_op,_ = is_market_open(symbol)
        p,c,pct = get_live_data(symbol)
        clr = "#00e676" if c>=0 else "#ff1744"
        dot = "🟢" if is_op else "🔴"
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {clr};">
                <div style="font-size:12px; color:#888;">{dot} {name}</div>
                <div style="font-size:26px; font-weight:bold;">₹{p:,.2f}</div>
                <div style="color:{clr}; font-weight:bold;">{c:+.2f} ({pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📰 Top Market Headlines")
    news = get_news("Indian Stock Market")
    for n in news:
        st.markdown(f'<div class="news-card"><div style="font-size:11px; color:#aaa;">{n["source"]} • {n["date"]}</div><a href="{n["link"]}" target="_blank" style="color:white; font-weight:bold; text-decoration:none;">{n["title"]}</a><div style="margin-top:5px; font-size:12px;">{n["sent"]}</div></div>', unsafe_allow_html=True)

# --------------------------
# 14. TOP 5 AI PICKS
if view=="⭐ Top 5 AI Picks":
    st.title("⭐ Top 5 AI Picks for Tomorrow")
    ai_results=[]
    st.info("Computing top AI picks. This may take a minute...")
    for name,symbol in NIFTY_100_TICKERS.items():
        df_s = yf.download(symbol, period="3mo", interval="1d")
        if df_s.empty: continue
        df_s = add_indicators(df_s)
        pred,_ = train_ai(df_s)
        if pred:
            curr,_ = get_live_data(symbol)
            diff = pred - curr
            ai_results.append((name,symbol,diff))
    if ai_results:
        top5 = sorted(ai_results,key=lambda x:x[2],reverse=True)[:5]
        for name,symbol,diff in top5:
            curr,_,_ = get_live_data(symbol)
            sig = "BUY 🚀" if diff>0 else "SELL 🔻"
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #2962ff;">
                <div style="font-size:16px; font-weight:bold;">{name} ({symbol})</div>
                <div style="font-size:26px; font-weight:bold
