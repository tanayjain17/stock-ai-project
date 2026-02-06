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
from datetime import datetime
import time

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
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid #2962ff;
        transition: transform 0.2s;
    }
    .news-card:hover {
        transform: translateX(5px);
        background-color: #1a1f2b;
    }
    
    /* Custom Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #12151e;
        border-right: 1px solid #2a2f3d;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ASSET LISTS (EXPANDED TO NIFTY 100 & ETFs) ---

# Top 100 Indian Stocks (Nifty 100) - Manually Curated for Speed
NIFTY_100_TICKERS = {
    # Top Giants
    "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS", "L&T": "LT.NS",
    "Kotak Bank": "KOTAKBANK.NS", "Axis Bank": "AXISBANK.NS", "HUL": "HINDUNILVR.NS",
    
    # Auto & Metals
    "Tata Motors": "TATAMOTORS.NS", "Maruti Suzuki": "MARUTI.NS", "M&M": "M&M.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS", "Eicher Motors": "EICHERMOT.NS", "Hero MotoCorp": "HEROMOTOCO.NS",
    "Tata Steel": "TATASTEEL.NS", "JSW Steel": "JSWSTEEL.NS", "Hindalco": "HINDALCO.NS",
    "Vedanta": "VEDL.NS", "Coal India": "COALINDIA.NS",
    
    # Finance & Tech
    "Bajaj Finance": "BAJFINANCE.NS", "Bajaj Finserv": "BAJAJFINSV.NS", "Jio Financial": "JIOFIN.NS",
    "HCL Tech": "HCLTECH.NS", "Wipro": "WIPRO.NS", "Tech Mahindra": "TECHM.NS", "LTIMindtree": "LTIM.NS",
    
    # Pharma & Energy
    "Sun Pharma": "SUNPHARMA.NS", "Dr Reddys": "DRREDDY.NS", "Cipla": "CIPLA.NS", "Divis Lab": "DIVISLAB.NS",
    "NTPC": "NTPC.NS", "Power Grid": "POWERGRID.NS", "ONGC": "ONGC.NS", "Adani Green": "ADANIGREEN.NS",
    "Adani Power": "ADANIPOWER.NS", "Tata Power": "TATAPOWER.NS",
    
    # Consumer & Others
    "Asian Paints": "ASIANPAINT.NS", "Titan": "TITAN.NS", "Nestle India": "NESTLEIND.NS",
    "Zomato": "ZOMATO.NS", "Paytm": "PAYTM.NS", "PB Fintech (PolicyBazaar)": "POLICYBZR.NS",
    "Adani Enterprises": "ADANIENT.NS", "Adani Ports": "ADANIPORTS.NS", "UltraTech Cement": "ULTRACEMCO.NS",
    "Grasim": "GRASIM.NS", "Ambuja Cements": "AMBUJACEM.NS", "DLF": "DLF.NS",
    "HAL": "HAL.NS", "BEL": "BEL.NS", "Mazagon Dock": "MAZDOCK.NS", "IRFC": "IRFC.NS", "RVNL": "RVNL.NS"
}

ETFS_MFS = {
    "Nifty 50 ETF (NIFTYBEES)": "NIFTYBEES.NS",
    "Bank Nifty ETF (BANKBEES)": "BANKBEES.NS",
    "IT Tech ETF (ITBEES)": "ITBEES.NS",
    "Pharma ETF (PHARMABEES)": "PHARMABEES.NS",
    "Gold ETF (GOLDBEES)": "GOLDBEES.NS",
    "Silver ETF (SILVERBEES)": "SILVERBEES.NS",
    "US Nasdaq 100 (MON100)": "MON100.NS",
    "HDFC Sensex ETF": "HDFCSENSEX.NS",
    "CPSE ETF (Govt Stocks)": "CPSEETF.NS",
    "PSU Bank ETF": "PSUBNKBEES.NS"
}

COMMODITIES_GLOBAL = {
    "Gold (Global Spot)": "GC=F",
    "Silver (Global Spot)": "SI=F",
    "Crude Oil (WTI)": "CL=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F"
}

# --- 3. HELPER FUNCTIONS ---

def get_live_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.fast_info.last_price
        prev_close = stock.fast_info.previous_close
        change = price - prev_close
        pct = (change / prev_close) * 100
        return price, change, pct
    except:
        return 0.0, 0.0, 0.0

def get_news_with_sources(query_term):
    try:
        # 1. Fetch Google News
        safe_query = urllib.parse.quote(query_term)
        rss_url = f"https://news.google.com/rss/search?q={safe_query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(rss_url)
        
        news_items = []
        for entry in feed.entries:
            title = entry.title
            link = entry.link
            source = entry.source.title if 'source' in entry else "Google News"
            
            # 2. Date Parsing & Sorting Logic
            # feedparser usually provides 'published_parsed' (time struct)
            pub_date = entry.get("published", str(datetime.now()))
            try:
                # Convert struct_time to comparable timestamp for sorting
                timestamp = time.mktime(entry.published_parsed) if 'published_parsed' in entry else 0
            except:
                timestamp = 0
            
            # Sentiment
            blob = TextBlob(title)
            pol = blob.sentiment.polarity
            sentiment = "🟢 Bullish" if pol > 0.05 else "🔴 Bearish" if pol < -0.05 else "⚪ Neutral"
            
            news_items.append({
                "title": title, 
                "link": link, 
                "source": source, 
                "date_str": pub_date[:16], # Display string
                "timestamp": timestamp,    # Sorting key
                "sentiment": sentiment
            })
            
        # 3. SORT: Newest First (Reverse Timestamp)
        news_items.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return news_items[:10] # Return top 10 sorted
    except Exception as e:
        return []

def add_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    return df

# --- 4. SIDEBAR CONFIG ---
st.sidebar.title("🔍 Market Scanner")

# View Mode
view_mode = st.sidebar.radio("Navigation", ["🏠 Market Dashboard", "📈 Stock Analyzer", "🏦 ETFs & Funds", "🛢️ Global Commodities"])

selected_ticker = "RELIANCE.NS" # Default

if view_mode == "📈 Stock Analyzer":
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Select Stock")
    
    # Nifty 100 Dropdown
    stock_name = st.sidebar.selectbox("Top 100 Stocks", list(NIFTY_100_TICKERS.keys()))
    selected_ticker = NIFTY_100_TICKERS[stock_name]
    
    # Exchange Toggle
    ex_toggle = st.sidebar.radio("Exchange", ["NSE", "BSE"], horizontal=True, index=0)
    if ex_toggle == "BSE":
        selected_ticker = selected_ticker.replace(".NS", ".BO")
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. OR Search Any Stock")
    custom_search = st.sidebar.text_input("Type Ticker (e.g. IRFC, ZOMATO)")
    if custom_search:
        # Intelligent suffix adding
        clean_search = custom_search.upper().replace(".NS","").replace(".BO","")
        selected_ticker = f"{clean_search}.NS" if ex_toggle == "NSE" else f"{clean_search}.BO"

elif view_mode == "🏦 ETFs & Funds":
    st.sidebar.subheader("Select ETF / Mutual Fund")
    etf_name = st.sidebar.selectbox("Popular ETFs", list(ETFS_MFS.keys()))
    selected_ticker = ETFS_MFS[etf_name]

elif view_mode == "🛢️ Global Commodities":
    st.sidebar.subheader("Select Commodity")
    comm_name = st.sidebar.selectbox("Asset", list(COMMODITIES_GLOBAL.keys()))
    selected_ticker = COMMODITIES_GLOBAL[comm_name]


# Global Params (Charts & AI)
if view_mode != "🏠 Market Dashboard":
    st.sidebar.markdown("---")
    st.sidebar.caption("Analysis Settings")
    chart_range = st.sidebar.selectbox("Chart History", ["1 Day", "1 Week", "1 Month", "6 Months", "1 Year", "5 Years"], index=2)
    ai_horizon = st.sidebar.selectbox("AI Prediction", ["Next 15 Mins", "Next 1 Hour", "Next 1 Day"])


# --- 5. PAGE 1: MARKET DASHBOARD ---
if view_mode == "🏠 Market Dashboard":
    st.title("🌏 Indian Markets at a Glance")
    st.markdown("Real-time updates from NSE & BSE")
    
    col1, col2, col3, col4 = st.columns(4)
    indices = [("NIFTY 50", "^NSEI", col1), ("SENSEX", "^BSESN", col2), 
               ("BANK NIFTY", "^NSEBANK", col3), ("INDIA VIX", "^INDIAVIX", col4)]
    
    for name, sym, col in indices:
        p, c, pct = get_live_price(sym)
        clr = "#00e676" if c >= 0 else "#ff1744"
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {clr};">
                <div style="color:#aaa; font-size:14px;">{name}</div>
                <div style="font-size:24px; font-weight:bold; color:#fff;">{p:,.2f}</div>
                <div style="color:{clr}; font-weight:bold;">{c:+.2f} ({pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader("📰 Latest Market News (Live)")
    
    # News Feed
    news = get_news_with_sources("Indian Stock Market Nifty Sensex")
    for n in news:
        st.markdown(f"""
        <div class="news-card">
            <div style="display:flex; justify-content:space-between; font-size:12px; color:#888; margin-bottom:5px;">
                <span>📢 {n['source']}</span>
                <span>🕒 {n['date_str']}</span>
            </div>
            <a href="{n['link']}" target="_blank" style="color:#e0e0e0; text-decoration:none; font-size:16px; font-weight:600;">
                {n['title']}
            </a>
            <div style="margin-top:5px; font-size:12px; font-weight:bold;">{n['sentiment']}</div>
        </div>
        """, unsafe_allow_html=True)

# --- 6. PAGE 2/3/4: ANALYSIS ENGINE ---
else:
    st.title(f"⚡ {selected_ticker} Analysis")
    
    # Live Header
    lp, lc, lpct = get_live_price(selected_ticker)
    l_clr = "#00e676" if lc >= 0 else "#ff1744"
    
    st.markdown(f"""
    <div style="background:#1e2330; padding:20px; border-radius:12px; border:1px solid #333; display:flex; align-items:center; gap:20px;">
        <div>
            <div style="color:#888; font-size:12px;">LIVE PRICE</div>
            <div style="font-size:42px; font-weight:bold; color:{l_clr};">₹{lp:,.2f}</div>
        </div>
        <div style="background:{l_clr}15; color:{l_clr}; padding:5px 15px; border-radius:15px; font-weight:bold;">
            {lc:+.2f} ({lpct:+.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    if st.button("🚀 Run Deep Analysis"):
        with st.spinner(f"Analyzing {selected_ticker}..."):
            try:
                # Map timeframe to yfinance params
                p_map = {"1 Day": "1d", "1 Week": "5d", "1 Month": "1mo", "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"}
                period = p_map[chart_range]
                interval = "5m" if period in ["1d", "5d"] else "1d"
                
                df = yf.download(selected_ticker, period=period, interval=interval, progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                
                if not df.empty:
                    df = add_indicators(df)
                    
                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
                    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AI Core
                    st.subheader("🤖 AI Prediction Model")
                    
                    # Data Prep
                    dataset = df[['Close']].values
                    scaler = MinMaxScaler(feature_range=(0,1))
                    scaled_data = scaler.fit_transform(dataset)
                    
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
                        model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
                        
                        # Predict
                        last_60 = scaled_data[-60:].reshape(1, 60, 1)
                        pred_val = scaler.inverse_transform(model.predict(last_60))[0][0]
                        
                        diff = pred_val - lp
                        direction = "BULLISH 🟢" if diff > 0 else "BEARISH 🔴"
                        
                        c1, c2 = st.columns(2)
                        c1.metric(f"Target ({ai_horizon})", f"₹{pred_val:.2f}")
                        c2.metric("Signal", direction, f"{diff:+.2f}")
                    else:
                        st.warning("Not enough data for AI prediction (Need >60 candles)")

                    # Sorted News
                    st.markdown("---")
                    st.subheader(f"📰 News: {selected_ticker}")
                    
                    # Clean ticker for news search (remove .NS)
                    search_term = selected_ticker.replace(".NS","").replace(".BO","")
                    stock_news = get_news_with_sources(search_term)
                    
                    if stock_news:
                        for sn in stock_news:
                            st.markdown(f"""
                            <div class="news-card">
                                <div style="display:flex; justify-content:space-between; font-size:12px; color:#aaa;">
                                    <span>🏛️ {sn['source']}</span>
                                    <span>{sn['date_str']}</span>
                                </div>
                                <a href="{sn['link']}" target="_blank" style="color:white; text-decoration:none; font-weight:bold; font-size:15px; display:block; margin-top:5px;">
                                    {sn['title']}
                                </a>
                                <div style="margin-top:5px; font-size:12px; font-weight:bold;">{sn['sentiment']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No specific news found.")

                else:
                    st.error("Could not fetch data. Market might be closed.")
            except Exception as e:
                st.error(f"Error: {e}")
