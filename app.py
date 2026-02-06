import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# 1. SETUP PAGE
st.set_page_config(page_title="Super Stock AI", layout="wide")
st.title("📈 Super Stock AI: Analysis + Prediction")

# --- INPUT SECTION (Top Bar) ---
col1, col2 = st.columns([1, 2]) # Make the second column wider for the buttons

with col1:
    ticker = st.text_input("Ticker (e.g., RELIANCE.NS, TCS.NS)", "RELIANCE.NS")

with col2:
    st.write("Select Time Range:")
    # This creates the horizontal buttons like TradingView
    time_period = st.radio(
        "", 
        ["3mo", "6mo", "1y", "3y", "5y", "max"], 
        index=2, # Default to '1y'
        horizontal=True,
        format_func=lambda x: x.upper() # Makes '1y' look like '1Y'
    )

# 2. TECHNICAL INDICATORS FUNCTION
def add_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# 3. MAIN LOGIC
if st.button("Analyze & Predict"):
    st.write(f"Fetching data for **{ticker}**...")
    
    # DOWNLOAD LOGIC:
    # If the user selects a short period (like 3mo), we might not have enough 
    # data for the AI (which needs 60 days context + training history).
    # So we ALWAYS download at least 1 year for calculations, 
    # but we will 'slice' the chart to show only what the user asked for.
    
    download_period = time_period
    if time_period in ["3mo", "6mo"]:
        download_period = "1y" # Force at least 1y download for AI stability
        
    data = yf.download(ticker, period=download_period, interval="1d")
    
    if len(data) > 60:
        # --- PART A: DASHBOARD ---
        data = add_indicators(data)
        
        # SLICE DATA FOR CHART VIEW (Based on user selection)
        # If user wanted 3mo, we only show the last ~90 rows in the chart
        if time_period == "3mo":
            chart_data = data.tail(90)
        elif time_period == "6mo":
            chart_data = data.tail(180)
        else:
            chart_data = data
            
        # --- NEW: FUNDAMENTALS ---
        st.markdown("### 🏢 Company Fundamentals")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            pe_ratio = info.get('trailingPE', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            if market_cap != 'N/A':
                # Smart format for Indian vs US stocks
                if str(ticker).endswith(".NS"):
                    market_cap = f"₹{market_cap / 10000000:.2f} Cr" 
                else:
                    market_cap = f"${market_cap / 1000000000:.2f} B"
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Cap", market_cap)
            c2.metric("P/E Ratio", pe_ratio)
            c3.metric("52W High", info.get('fiftyTwoWeekHigh', 'N/A'))
            c4.metric("52W Low", info.get('fiftyTwoWeekLow', 'N/A'))
        except:
            st.warning("Could not fetch fundamentals (common for indices/crypto).")
            
        st.markdown("---")

        # --- PLOT CHART ---
        st.subheader(f"Price Chart ({time_period.upper()})")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=chart_data.index,
                        open=chart_data['Open'], high=chart_data['High'],
                        low=chart_data['Low'], close=chart_data['Close'], name='OHLC'))
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA_50'], line=dict(color='orange', width=1), name='50-Day SMA'))
        st.plotly_chart(fig, use_container_width=True)
        
        # --- PART B: AI TRAINING ---
        st.markdown("---")
        st.subheader("🤖 AI Prediction")
        
        progress_bar = st.progress(0)
        st.write("Training model on live data...")
        
        # Data Prep
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
        
        progress_bar.progress(40)
        
        # Train
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=3, verbose=0)
        
        progress_bar.progress(80)
        
        # Predict
        last_60_days = scaled_data[-60:]
        X_test = []
        X_test.append(last_60_days)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        pred_price = model.predict(X_test)
        pred_price_unscaled = scaler.inverse_transform(pred_price)
        final_prediction = pred_price_unscaled[0][0]
        
        progress_bar.progress(100)
        
        # Result
        current_price = data['Close'].iloc[-1].item()
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{current_price:.2f}")
        c2.metric("AI Prediction (Next Day)", f"{final_prediction:.2f}", 
                    delta=f"{final_prediction - current_price:.2f}")
            
    else:
        st.error("Not enough data to analyze! Try a longer time period.")
