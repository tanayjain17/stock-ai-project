import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go

# 1. SETUP
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Cloud Stock Predictor")

# 2. LOAD RESOURCES (Cached for speed)
@st.cache_resource
def load_resources():
    model = load_model("stock_model.h5")
    scaler = joblib.load("stock_scaler.gz")
    return model, scaler

try:
    model, scaler = load_resources()
    st.success("✅ AI Model Loaded from Cloud")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. USER INPUT
ticker = st.text_input("Enter Ticker (Use exact Yahoo Finance code, e.g., RELIANCE.NS)", "RELIANCE.NS")

if st.button("Predict"):
    # 4. FETCH DATA
    data = yf.download(ticker, period="1y", interval="1d")
    
    if len(data) > 60:
        # 5. PREPARE DATA FOR AI
        # We need the last 60 days of data to predict the next day
        last_60_days = data['Close'].values[-60:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 6. PREDICT
        pred_price = model.predict(X_test)
        pred_price_unscaled = scaler.inverse_transform(pred_price)
        
        # 7. SHOW RESULTS
        current_price = data['Close'].iloc[-1].item() # Extract float value
        predicted_val = pred_price_unscaled[0][0]
        
        col1, col2 = st.columns(2)
        col1.metric("Current Price", f"₹{current_price:.2f}")
        col2.metric("AI Prediction (Next Day)", f"₹{predicted_val:.2f}", delta=f"{predicted_val - current_price:.2f}")
        
        # 8. CHART
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
        st.plotly_chart(fig)
        
    else:
        st.error("Not enough data to make a prediction (Need at least 60 days).")
