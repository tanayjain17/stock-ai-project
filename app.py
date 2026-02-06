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

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Market Pulse AI", layout="wide", page_icon="📈")

st.markdown("""
<style>
    /* Modern Dark Theme */
    .stApp { background-color: #0e1117; color: white; }
    
    /* Card Styles */
    .metric-card {
        background-color: #1e2330;
        border: 1px solid #2a2f3d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* News Feed Styles */
    .news-card {
        background-color: #151922;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
        border-left: 4px solid #2962ff;
    }
    
    /* Ticker Tape Animation */
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ASSET LISTS (The "Master Database") ---
# Top 50 Indian Stocks (Nifty 50) + Popular F/O
NIFTY_50_TICKERS = {
    "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS", "L&T": "LT.NS",
    "Kotak Bank": "KOTAKBANK.NS", "Axis Bank": "AXISBANK.NS", "HUL": "HINDUNILVR.NS",
    "Tata Motors": "TATAMOTORS.NS", "Maruti Suzuki": "MARUTI.NS", "Asian Paints": "ASIANPAINT.NS",
    "Sun Pharma": "SUNPHARMA.NS", "Titan": "TITAN.NS", "Bajaj Finance": "BAJFINANCE.NS",
    "UltraTech Cement": "ULTRACEMCO.NS", "ONGC": "ONGC.NS", "NTPC": "NTPC.NS",
    "Wipro": "WIPRO.NS", "Power Grid": "POWERGRID.NS", "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS", "Adani Enterprises": "ADANIENT.NS", "Adani Ports": "ADANIPORTS.NS",
    "Coal India": "COALINDIA.NS", "M&M": "M&M.NS", "HCL Tech": "HCLTECH.NS"
}

INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX"
}

COMMODITIES = {
    "Gold (Global)": "GC=F",
    "Silver (Global)": "SI=F",
    "Crude Oil (WTI)": "CL=F",
    "Gold Bees (India)": "GOLDBEES.NS",
    "Silver Bees (India)": "SILVERBEES.NS"
}

# --- 3. HELPER FUNCTIONS ---

def get_live_price(symbol):
    try:
        # Ultra-fast fetch for current price only
        stock = yf.Ticker(symbol)
        price = stock.fast_info.last_price
        prev_close = stock.fast_info.previous_close
        
        # Calculate change
        change = price - prev_close
        pct = (change / prev_close) * 100
        return price, change, pct
    except:
        return 0.0, 0.0, 0.0

def get_news_with_sources(query_term):
    try:
        # Using Google News RSS which includes source info
        safe_query = urllib.parse.quote(query_term)
        rss_url = f"https://news.google.com/rss/search?q={safe_query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        
        news_items = []
        for entry in feed.entries[:10]: # Increased to 10 items
            title = entry.title
            link = entry.link
            source = entry.source.title if 'source' in entry else "Google News"
            published = entry.published[:16] # Shorten date
            
            # Sentiment Analysis
            blob = TextBlob(title)
            pol = blob.sentiment.polarity
            sentiment = "🟢 Bullish" if pol > 0.05 else "🔴 Bearish" if pol < -0.05 else "⚪ Neutral"
            
            news_items.append({
                "title": title, "link": link, "source": source,
                "date": published, "sentiment": sentiment
            })
        return news_items
    except:
        return []

def add_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    return df

# --- 4. SIDEBAR CONFIG ---
st.sidebar.title("🔍 Market Scanner")

# View Mode Selection
view_mode = st.sidebar.radio("Go to Section:", ["🏠 Market Dashboard", "📈 Stock Analysis", "🛢️ Commodities"])

if view_mode == "📈 Stock Analysis":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Asset")
    
    # "All Stocks" Logic: Combobox lets you type OR select
    selected_stock_name = st.sidebar.selectbox(
        "Search Stock (Type to search Nifty 50)", 
        list(NIFTY_50_TICKERS.keys()),
        index=0
    )
    
    # Exchange Toggle
    exchange = st.sidebar.radio("Exchange", ["NSE", "BSE"], horizontal=True)
    
    # Logic to handle ticker mapping
    base_ticker = NIFTY_50_TICKERS[selected_stock_name]
    ticker_symbol = base_ticker.replace(".NS", ".BO") if exchange == "BSE" else base_ticker
    
    # Custom Ticker Input (For stocks NOT in the list)
    st.sidebar.markdown("---")
    custom_ticker = st.sidebar.text_input("Or type ANY symbol (e.g., ZOMATO.NS):")
    if custom_ticker:
        ticker_symbol = custom_ticker.upper()

elif view_mode == "🛢️ Commodities":
    st.sidebar.subheader("Select Commodity")
    comm_name = st.sidebar.selectbox("Asset", list(COMMODITIES.keys()))
    ticker_symbol = COMMODITIES[comm_name]

# Global Params
if view_mode != "🏠 Market Dashboard":
    prediction_range = st.sidebar.selectbox("Prediction Horizon", ["Next 5 Mins", "Next 1 Hour", "Next 1 Day"])
    chart_period = st.sidebar.selectbox("Chart History", ["1d", "5d", "1mo", "6mo", "1y", "5y"])

# --- 5. PAGE 1: MARKET DASHBOARD (DEFAULT LANDING) ---
if view_mode == "🏠 Market Dashboard":
    st.title("🌏 Indian Markets at a Glance")
    st.markdown("Live updates from NSE & BSE Indices")
    
    # Fetch Index Data Live
    col1, col2, col3, col4 = st.columns(4)
    
    indices_list = [("NIFTY 50", "^NSEI", col1), ("SENSEX", "^BSESN", col2), 
                    ("BANK NIFTY", "^NSEBANK", col3), ("INDIA VIX", "^INDIAVIX", col4)]
    
    for name, sym, col in indices_list:
        price, chg, pct = get_live_price(sym)
        color = "green" if chg >= 0 else "red"
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {color};">
                <h4 style="margin:0; color: #aaa;">{name}</h4>
                <h2 style="margin:5px 0; color: white;">{price:,.2f}</h2>
                <p style="margin:0; color: {color}; font-weight:bold;">{chg:+.2f} ({pct:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader("📰 Top Market Headlines")
    
    # General Market News
    general_news = get_news_with_sources("Indian Stock Market Nifty Sensex")
    for news in general_news:
        st.markdown(f"""
        <div class="news-card">
            <div style="display:flex; justify-content:space-between; color:#aaa; font-size:12px;">
                <span>📢 {news['source']}</span>
                <span>🕒 {news['date']}</span>
            </div>
            <a href="{news['link']}" style="color:white; text-decoration:none; font-weight:600; font-size:16px; display:block; margin-top:5px;">
                {news['title']}
            </a>
        </div>
        """, unsafe_allow_html=True)

# --- 6. PAGE 2 & 3: ANALYSIS ENGINE ---
else:
    st.title(f"⚡ {ticker_symbol} Analysis")
    
    # 1. LIVE HEADER
    l_price, l_chg, l_pct = get_live_price(ticker_symbol)
    l_color = "#00e676" if l_chg >= 0 else "#ff1744"
    
    st.markdown(f"""
    <div style="padding: 20px; background: #1e2330; border-radius: 10px; display: flex; align-items: center; gap: 20px;">
        <div>
            <span style="font-size: 14px; color: #aaa;">CURRENT PRICE</span>
            <div style="font-size: 36px; font-weight: bold; color: {l_color};">₹{l_price:,.2f}</div>
        </div>
        <div style="background: {l_color}20; padding: 5px 15px; border-radius: 20px; color: {l_color}; font-weight: bold;">
            {l_chg:+.2f} ({l_pct:+.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    if st.button("🔄 Run Deep Analysis"):
        with st.spinner("Crunching numbers..."):
            try:
                # Get Chart Data
                data = yf.download(ticker_symbol, period=chart_period if 'd' not in chart_period else "5d", interval="1d" if 'y' in chart_period else "5m")
                
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)

                if not data.empty:
                    data = add_indicators(data)
                    
                    # PLOTLY CHART
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"))
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='orange', width=1.5), name="50 SMA"))
                    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AI PREDICTION (Simulated for speed/stability in this snippet)
                    st.subheader("🤖 AI Price Target")
                    
                    # Simple LSTM Logic (Simplified for robustness)
                    df_ai = data[['Close']].values
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(df_ai)
                    
                    if len(scaled_data) > 60:
                        x_train, y_train = [], []
                        for i in range(60, len(scaled_data)):
                            x_train.append(scaled_data[i-60:i, 0])
                            y_train.append(scaled_data[i, 0])
                        
                        x_train, y_train = np.array(x_train), np.array(y_train)
                        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                        
                        model = Sequential()
                        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                        model.add(LSTM(50, return_sequences=False))
                        model.add(Dense(25))
                        model.add(Dense(1))
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0) # Fast epoch
                        
                        last_60 = scaled_data[-60:].reshape(1, 60, 1)
                        pred = scaler.inverse_transform(model.predict(last_60))[0][0]
                        
                        diff = pred - l_price
                        signal = "BUY 🚀" if diff > 0 else "SELL 🔻"
                        
                        col_a, col_b = st.columns(2)
                        col_a.metric(f"AI Prediction ({prediction_range})", f"₹{pred:.2f}")
                        col_b.metric("Signal Strength", signal, f"{diff:+.2f}")
                    else:
                        st.warning("Not enough data points for AI analysis.")

                    # NEWS SECTION
                    st.markdown("---")
                    st.subheader(f"📰 News for {ticker_symbol}")
                    
                    news_list = get_news_with_sources(ticker_symbol.replace(".NS", "").replace(".BO", ""))
                    if news_list:
                        for item in news_list:
                            st.markdown(f"""
                            <div class="news-card">
                                <div style="display:flex; justify-content:space-between; font-size:12px; color:#888;">
                                    <span>🏛️ {item['source']}</span>
                                    <span>{item['sentiment']}</span>
                                </div>
                                <a href="{item['link']}" target="_blank" style="text-decoration:none; color:#eee; font-weight:bold; font-size:15px;">
                                    {item['title']}
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No specific news found.")
                else:
                    st.error("Data fetch failed. Ticker might be invalid or market closed.")
            except Exception as e:
                st.error(f"Analysis Error: {e}")
