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

# 2. SIDEBAR INPUT
ticker = st.sidebar.text_input("Enter Ticker (e.g., RELIANCE.NS, TCS.NS)", "RELIANCE.NS")
days_history = st.sidebar.slider("Days of data to train on:", 365, 2000, 1000)

# 3. TECHNICAL INDICATORS FUNCTION
def add_indicators(df):
    # SMA (Simple Moving Average)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# 4. MAIN LOGIC
if st.sidebar.button("Analyze & Predict"):
    st.write(f"Fetching data for **{ticker}**...")
    
    # Fetch Data
    data = yf.download(ticker, period="5y", interval="1d")
    
    if len(data) > 0:
        # --- PART A: DASHBOARD (Indicators) ---
        data = add_indicators(data)
        
        # Show Price Chart
        st.subheader(f"Price Chart: {ticker}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'], name='OHLC'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='orange', width=1), name='50-Day SMA'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Show RSI
        st.subheader("RSI Indicator (Overbought > 70, Oversold < 30)")
        st.line_chart(data['RSI'].tail(100))
        
        # --- PART B: UNIVERSAL AI TRAINING (On the Fly) ---
        st.markdown("---")
        st.subheader("🤖 AI Prediction")
        st.write("Training a custom AI model for this stock... (This takes ~30 seconds)")
        
        progress_bar = st.progress(0)
        
        # 1. Prepare Data
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
        
        progress_bar.progress(30)
        
        # 2. Build & Train Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Fast training (3 epochs is enough for a quick web demo)
        model.fit(x_train, y_train, batch_size=1, epochs=3, verbose=0)
        
        progress_bar.progress(80)
        
        # 3. Predict Tomorrow
        last_60_days = scaled_data[-60:]
        X_test = []
        X_test.append(last_60_days)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        pred_price = model.predict(X_test)
        pred_price_unscaled = scaler.inverse_transform(pred_price)
        final_prediction = pred_price_unscaled[0][0]
        
        progress_bar.progress(100)
        
        # 4. Display Result
        current_price = data['Close'].iloc[-1].item()
        
        col1, col2 = st.columns(2)
        col1.metric("Current Price", f"₹{current_price:.2f}")
        col2.metric("AI Predicted Price (Next Day)", f"₹{final_prediction:.2f}", 
                    delta=f"{final_prediction - current_price:.2f}")
                    
        if final_prediction > current_price:
            st.success("🚀 AI Sentiment: BULLISH")
        else:
            st.error("🔻 AI Sentiment: BEARISH")
            
    else:
        st.error("No data found! Check the ticker symbol.")
