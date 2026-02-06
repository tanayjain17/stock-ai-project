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

# --- NEWS SENTIMENT FUNCTION (FIXED WITH FEEDPARSER) ---
def get_news_sentiment(ticker_symbol):
    try:
        # Use the official Yahoo Finance RSS feed for the specific ticker
        # This is 100x more reliable than yfinance.Ticker.news
        rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker_symbol}'
        feed = feedparser.parse(rss_url)
        
        news_data = []
        for entry in feed.entries[:5]: # Get top 5
            title = entry.title
            link = entry.link
            published = entry.published
            
            # Analyze Sentiment
            analysis = TextBlob(title)
            polarity = analysis.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = "🟢 Positive"
            elif polarity < -0.1:
                sentiment = "🔴 Negative"
            else:
                sentiment = "⚪ Neutral"
                
            news_data.append({
                'Title': title, 
                'Link': link, 
                'Sentiment': sentiment,
                'Date': published
            })
        return news_data
    except Exception as e:
        return []

# 3. MAIN LOGIC
if st.sidebar.button("Analyze & Predict"):
    
    interval, ai_period, view_period = get_data_params(prediction_option, chart_view)
    st.write(f"Fetching **{interval}** data for **{ticker}**...")
    
    try:
        data = yf.download(ticker, period=ai_period, interval=interval)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        if len(data) > 0:
            data = fix_timezone(data)
            
            # --- FUNDAMENTALS ---
            st.markdown("### 🏢 Company Fundamentals")
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                pe_ratio = info.get('trailingPE', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                
                if market_cap != 'N/A':
                    if str(ticker).endswith(".NS"):
                        market_cap = f"₹{market_cap / 10000000:.2f} Cr" 
                    else:
                        market_cap = f"${market_cap / 1000000000:.2f} B"
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Market Cap", market_cap)
                c2.metric("P/E Ratio", pe_ratio)
                c3.metric("52W High", info.get('fiftyTwoWeekHigh', 'N/A'))
                c4.metric("Day Low/High", f"{info.get('dayLow','-')} / {info.get('dayHigh','-')}")
            except:
                st.warning("Could not fetch fundamentals.")
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
                curr_price = data['Close'].iloc[-1].item()
                
                if interval == '1d':
                    last_time = data.index[-1].strftime('%d-%b-%Y')
                else:
                    last_time = data.index[-1].strftime('%H:%M %p')
                
                diff = final_val - curr_price
                pct = (diff / curr_price) * 100
                color = "green" if diff > 0 else "red"
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Last Price", f"₹{curr_price:.2f}", f"Date: {last_time}")
                col2.metric(f"Prediction ({prediction_option})", f"₹{final_val:.2f}")
                col3.write(f"### Move: :{color}[{pct:.2f}%]")
                
                if diff > 0:
                    st.success(f"🚀 AI Signal: BULLISH (Up by ₹{diff:.2f})")
                else:
                    st.error(f"🔻 AI Signal: BEARISH (Down by ₹{abs(diff):.2f})")
            else:
                st.warning("⚠️ Not enough data for AI prediction.")
            
            # --- NEWS SECTION (FIXED) ---
            st.markdown("---")
            st.markdown("### 📰 Latest News & Sentiment")
            
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
