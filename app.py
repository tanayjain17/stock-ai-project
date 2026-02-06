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
        # Force fetch 1-minute data for the last 1 day to get the true live tick
        live_data = yf.download(symbol, period="1d", interval="1m", progress=False)
        
        if not live_data.empty:
            if live_data.index.tz is None:
                live_data.index = live_data.index.tz_localize('UTC')
            live_data.index = live_data.index.tz_convert('Asia/Kolkata')
            
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
            
        # 2. GET SEPARATE REAL-TIME PRICE (The "Truth" Source)
        real_price, real_time = get_real_time_price(ticker)
        
        if len(data) > 0:
            data = fix_timezone(data)
            
            # --- DASHBOARD HEADER ---
            st.markdown("### 📊 Market Dashboard")
            
            # Use the REAL-TIME fetch if available, otherwise fallback to chart data
            display_price = real_price if real_price else data['Close'].iloc[-1]
            display_time = real_time if real_time else data.index[-1]
            
            # Format time string
            time_str = display_time.strftime('%H:%M:%S') if display_time else "--:--"
            
            # Fetch Prev Close for comparison
            stock = yf.Ticker(ticker)
            prev_close = stock.fast_info.get('previous_close', display_price)
            
            # Calculate Change
            change = display_price - prev_close
            pct_change = (change / prev_close) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Live Price", f"₹{display_price:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
            c2.metric("Last Update", f"{time_str}")
            
            # Market Cap (Safe Fetch)
            mkt_cap = stock.fast_info.get('market_cap')
            if mkt_cap:
                val = f"₹{mkt_cap/1e7:.0f} Cr" if ".NS" in ticker else f"${mkt_cap/1e9:.2f} B"
                c3.metric("Market Cap", val)
            else:
                c3.metric("Market Cap", "N/A")
            
            st.markdown("---")

            # --- PLOT CHART ---
            data = add_indicators(data)
            
            if chart_view == "1 Day": chart_data = data.tail(75)
            elif chart_view == "5 Days": chart_data = data.tail(375)
            else: chart_data = data
            
            st.subheader(f"Price Chart ({ticker})")
            fig = go.Figure()
            
            date_fmt = '%Y-%m-%d' if interval == '1d' else '%Y-%m-%d %H:%M'
            
            fig.add_trace(go.Candlestick(
                x=chart_data.index.strftime(date_fmt),
                open=chart_data['Open'], high=chart_data['High'],
                low=chart_data['Low'], close=chart_data['Close'], name='OHLC'
            ))
            fig.add_trace(go.Scatter(
                x=chart_data.index.strftime(date_fmt), 
                y=chart_data['SMA_50'], line=dict(color='orange', width=1), name='50 Period SMA'
            ))
            fig.update_layout(xaxis_type='category', xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # --- AI TRAINING & PREDICTION ---
            if len(data) > 60:
                st.markdown("### 🤖 Artificial Intelligence")
                
                progress = st.progress(0)
                status = st.empty()
                status.write(f"Training 'Brain' on {interval} candles...")
                
                # Prep
                df_ai = data[['Close']].values
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(df_ai)
                
                x_train, y_train = [], []
                time_step = 60
                
                for i in range(time_step, len(scaled_data)):
                    x_train.append(scaled_data[i-time_step:i, 0])
                    y_train.append(scaled_data[i, 0])
                
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                
                progress.progress(40)
                
                # Model
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, batch_size=1, epochs=3, verbose=0)
                
                progress.progress(80)
                status.write("Calculating Probability...")
                
                # Predict
                last_60 = scaled_data[-60:]
                X_test = np.array([last_60])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                
                pred_price = model.predict(X_test)
                pred_price_unscaled = scaler.inverse_transform(pred_price)
                final_val = pred_price_unscaled[0][0]
                
                progress.progress(100)
                status.empty()
                
                # Results Display
                diff = final_val - display_price # Use the REAL live price
                pct = (diff / display_price) * 100
                color = "green" if diff > 0 else "red"
                
                col1, col2 = st.columns(2)
                col1.metric(f"AI Target ({prediction_option})", f"₹{final_val:.2f}")
                col2.write(f"### Signal: :{color}[{pct:.2f}%]")
                
                if diff > 0:
                    st.success(f"🚀 AI Signal: BULLISH")
                else:
                    st.error(f"🔻 AI Signal: BEARISH")
            else:
                st.warning("⚠️ Not enough data for AI prediction.")
            
            # --- NEWS SECTION ---
            st.markdown("---")
            st.markdown("### 📰 Latest News & Sentiment (Google News)")
            
            news_items = get_news_sentiment(ticker)
            if news_items:
                for news in news_items:
                    col_s, col_t = st.columns([1, 4])
                    col_s.write(f"**{news['Sentiment']}**")
                    col_t.markdown(f"[{news['Title']}]({news['Link']})")
            else:
                st.info("No recent news found for this ticker.")
                
        else:
            st.error("No data found. Market might be closed or ticker is invalid.")
            
    except Exception as e:
        st.error(f"Error: {e}")
