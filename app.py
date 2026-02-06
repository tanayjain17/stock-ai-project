import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# 1. SETUP PAGE
st.set_page_config(page_title="Super Stock AI (Pro)", layout="wide")
st.title("⚡ Super Stock AI: Real-Time Pro")

# --- INPUT SECTION ---
col1, col2 = st.columns([1, 3])

with col1:
    ticker = st.text_input("Ticker (e.g., RELIANCE.NS, NVDA, BTC-USD)", "RELIANCE.NS")

with col2:
    st.write("Select Time Range:")
    time_period = st.radio(
        "", 
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"], 
        index=2, 
        horizontal=True,
        format_func=lambda x: x.upper()
    )

# --- HELPER: SMART INTERVAL SELECTOR ---
def get_interval(period):
    if period == "1d": return "5m"
    if period == "5d": return "15m"
    if period == "1mo": return "60m"
    if period == "3mo": return "1d"
    return "1d"

# 2. TECHNICAL INDICATORS
def add_indicators(df):
    window = 50 if len(df) > 50 else 20
    df['SMA'] = df['Close'].rolling(window=window).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# 3. MAIN LOGIC
if st.button("Analyze & Predict"):
    interval = get_interval(time_period)
    st.write(f"Fetching **{interval}** data for **{ticker}**...")
    
    try:
        data = yf.download(ticker, period=time_period, interval=interval)
        
        # Fix Multi-Index issue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        if len(data) > 0:
            # --- FUNDAMENTALS ---
            if time_period not in ["1d", "5d"]:
                st.markdown("### 🏢 Fundamentals")
                try:
                    info = yf.Ticker(ticker).info
                    market_cap = info.get('marketCap', 'N/A')
                    if market_cap != 'N/A':
                        market_cap = f"{market_cap / 1e7:.0f} Cr" if ".NS" in ticker else f"${market_cap / 1e9:.2f} B"
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Cap", market_cap)
                    c2.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                    c3.metric("52W High", info.get('fiftyTwoWeekHigh', 'N/A'))
                    st.markdown("---")
                except:
                    pass

            # --- PLOT CHART (FIXED) ---
            data = add_indicators(data)
            
            st.subheader(f"Price Chart ({time_period.upper()} | {interval} interval)")
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=data.index.strftime('%Y-%m-%d %H:%M'), # String format fixes the gaps
                            open=data['Open'], high=data['High'],
                            low=data['Low'], close=data['Close'], name='OHLC'))
            
            # SMA Line
            fig.add_trace(go.Scatter(x=data.index.strftime('%Y-%m-%d %H:%M'), 
                            y=data['SMA'], line=dict(color='orange', width=1), name='SMA'))
            
            # --- THE MAGIC FIX: REMOVE GAPS ---
            fig.update_layout(
                xaxis_type='category',   # This removes weekend/night gaps
                xaxis_rangeslider_visible=False, # Hides the bottom slider (cleaner look)
                xaxis_nticks=10 # Limits labels so they don't get messy
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # --- AI TRAINING ---
            if len(data) > 60:
                st.markdown("---")
                st.subheader(f"🤖 AI Prediction (Next {interval} candle)")
                
                progress = st.progress(0)
                status = st.empty()
                status.write("Normalizing data...")
                
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
                status.write("Training LSTM Model...")
                
                # Train
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, batch_size=1, epochs=3, verbose=0)
                
                progress.progress(80)
                status.write("Calculating prediction...")
                
                # Predict
                last_60 = scaled_data[-60:]
                X_test = np.array([last_60])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                
                pred_price = model.predict(X_test)
                pred_price_unscaled = scaler.inverse_transform(pred_price)
                final_val = pred_price_unscaled[0][0]
                
                progress.progress(100)
                status.empty()
                
                # Display Prediction
                curr_price = data['Close'].iloc[-1].item()
                c1, c2 = st.columns(2)
                c1.metric(f"Current Price ({interval})", f"{curr_price:.2f}")
                c2.metric(f"Predicted Next {interval}", f"{final_val:.2f}", 
                          delta=f"{final_val - curr_price:.2f}")

            else:
                st.warning(f"⚠️ Not enough data points to run AI (Need 60 candles, found {len(data)}).")

        else:
            st.error("No data found.")
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
