import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
from textblob import TextBlob
import feedparser
import urllib.parse
import datetime

# 1. SETUP PAGE
st.set_page_config(page_title="Super Stock AI (Pro)", layout="wide")
st.title("⚡ Super Stock AI: Real-Time Pro")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "RELIANCE.NS")

# PREDICTION TARGET
prediction_option = st.sidebar.selectbox(
    "🎯 AI Prediction Target",
    ("Next 5 Minutes", "Next 15 Minutes", "Next 30 Minutes", "Next 1 Hour", "Next 1 Day")
)

# CHART VIEW
chart_view = st.sidebar.selectbox(
    "👀 Chart History View",
    ("1 Day", "5 Days", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years")
)

# --- HELPER: TIMEZONE FIXER ---
def fix_timezone(df):
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('Asia/Kolkata')
    return df

# --- HELPER: FORCE REAL-TIME PRICE ---
def get_real_time_price(symbol):
    try:
        # Force fetch 1-minute data for the last 1 day
        # This bypasses the 'daily' cache and gets the true live tick
        live_data = yf.download(symbol, period="1d", interval="1m", progress=False)
        
        if not live_data.empty:
            # Fix timezone
            if live_data.index.tz is None:
                live_data.index = live_data.index.tz_localize('UTC')
            live_data.index = live_data.index.tz_convert('Asia/Kolkata')
            
            # Get the absolute last candle
            latest_price = live_data['Close'].iloc[-1].item()
            latest_time = live_data.index[-1]
            return latest_price, latest_time
        return None, None
    except:
        return None, None

# --- HELPER: DATA PARAMS ---
def get_data_params(predict_target, view_target):
    if predict_target == "Next 5 Minutes": interval = "5m"; period = "5d"
    elif predict_target == "Next 15 Minutes": interval = "15m"; period = "5d"
    elif predict_target == "Next 30 Minutes": interval = "30m"; period = "1mo"
    elif predict_target == "Next 1 Hour": interval = "60m"; period = "3mo"
    else: interval = "1d"; period = "5y"
    
    view_map = {
        "1 Day": "1d", "5 Days": "5d", "1 Month": "1mo", 
        "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"
    }
    requested_period = view_map[view_target]
    return interval, period, requested_period

# --- TECHNICAL INDICATORS ---
def add_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# --- NEWS SENTIMENT ---
def get_news_sentiment(ticker_symbol):
    try:
        query = urllib.parse.quote(ticker_symbol)
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        news_data = []
        if feed.entries:
            for entry in feed.entries[:5]: 
                title = entry.title
                link = entry.link
                analysis = TextBlob(title)
                polarity = analysis.sentiment.polarity
                if polarity > 0.05: sentiment = "🟢 Positive"
                elif polarity < -0.05: sentiment = "🔴 Negative"
                else: sentiment = "⚪ Neutral"
                news_data.append({'Title': title, 'Link': link, 'Sentiment': sentiment})
        return news_data
    except:
        return []

# 3. MAIN LOGIC
if st.sidebar.button("Analyze & Predict"):
    
    interval, ai_period, view_period = get_data_params(prediction_option, chart_view)
    st.write(f"Fetching **{interval}** data for **{ticker}**...")
    
    try:
        # 1. GET DATA FOR CHARTS & AI
        data = yf.download(ticker, period=ai_period, interval=interval)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        # 2. GET SEPARATE REAL-TIME PRICE (The "Truth
