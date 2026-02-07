import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
from datetime import datetime, time as dt_time
import pytz
import feedparser
from textblob import TextBlob
import urllib.parse
import time
import requests
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# --------------------------
# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Groww-Style Dashboard", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# --------------------------
# 2. SESSION STATE
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Market Dashboard"

def navigate_to(page_name):
    st.session_state.page = page_name

# --------------------------
# 3. MODERN "FUN" CSS
st.markdown("""
<style>
    .stApp { background-color: #0f1115; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }
    
    /* GLASSMORPHISM CARDS */
    .fun-card {
        background: rgba(30, 34, 45, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fun-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 208, 156, 0.15);
        border-color: #00d09c;
    }

    /* TEXT GRADIENTS */
    .gradient-text-green { background: -webkit-linear-gradient(45deg, #00d09c, #00ffaa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .gradient-text-red { background: -webkit-linear-gradient(45deg, #ff4b4b, #ff9068); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    
    /* HEADERS */
    .section-title { font-size: 22px; font-weight: 700; margin-top: 25px; margin-bottom: 15px; color: #e0e0e0; }

    /* NEWS */
    .news-box { padding: 12px; border-radius: 12px; background: #161920; margin-bottom: 10px; border-left: 3px solid #4c8bf5; }
    .news-link { color: #fff; text-decoration: none; font-weight: 500; font-size: 14px; }
    .news-link:hover { color: #4c8bf5; }
    .news-meta { font-size: 11px; color: #888; margin-top: 4px; }
    
    /* BUTTONS */
    .ext-btn { display: inline-block; padding: 8px 16px; margin: 5px; border-radius: 20px; background: #1E1E1E; border: 1px solid #333; color: white; text-decoration: none; font-size: 13px; transition: 0.2s; }
    .ext-btn:hover { border-color: #00d09c; color: #00d09c; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# 4. DATA POOLS
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}

# Most Traded Scanner Pool
SCANNER_POOL = [
    "ZOMATO.NS", "YESBANK.NS", "IDEA.NS", "TATASTEEL.NS", "RELIANCE.NS", 
    "HDFCBANK.NS", "SBIN.NS", "INFY.NS", "ITC.NS", "TATAMOTORS.NS",
    "ADANIENT.NS", "PNB.NS", "SUZLON.NS", "JPPOWER.NS", "NHPC.NS"
]

RSS_FEEDS = {
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "CNBC-TV18": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/stock-market.xml",
    "LiveMint": "https://www.livemint.com/rss/markets"
}

# --------------------------
# 5. SIDEBAR NAVIGATION
st.sidebar.title("🚀 Menu")
nav_options = ["🏠 Market Dashboard", "📈 Stock Analyzer", "🏦 ETFs & Mutual Funds", "🛢️ Global Commodities", "⭐ Top 5 AI Picks"]
selection = st.sidebar.radio("Navigate", nav_options, index=nav_options.index(st.session_state.page))

if selection != st.session_state.page:
    st.session_state.page = selection
    st.rerun()

view = st.session_state.page

# --------------------------
# 6. AUTO REFRESH
if view != "⭐ Top 5 AI Picks":
    st_autorefresh(interval=30000, key="dashboard_refresh")

# --------------------------
# 7. UTILITY FUNCTIONS
def get_live_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.fast_info
        price = info.last_price
        prev = info.previous_close
        if price is None or prev is None: return 0.0, 0.0, 0.0
        return price, price - prev, (price - prev) / prev * 100
    except:
        return 0.0, 0.0, 0.0

@st.cache_data(ttl=60)
def scan_most_traded():
    data = []
    for ticker in SCANNER_POOL:
        try:
            t = yf.Ticker(ticker)
            vol = t.fast_info.last_volume
            price = t.fast_info.last_price
            prev = t.fast_info.previous_close
            pct = ((price-prev)/prev)*100
            data.append({"symbol": ticker, "vol": vol, "price": price, "pct": pct})
        except: continue
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by='vol', ascending=False).head(4)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_news_multi():
    items = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:2]:
                ts = time.mktime(e.published_parsed) if 'published_parsed' in e else time.time()
                date_str = time.strftime('%H:%M', time.localtime(ts))
                items.append({'title':e.title, 'link':e.link, 'source':source, 'ts':ts, 'time':date_str})
        except: continue
    return sorted(items, key=lambda x: x['ts'], reverse=True)[:6]

# --------------------------
# 8. DASHBOARD VIEW
if view == "🏠 Market Dashboard":
    st.markdown('<div class="section-title">📊 Market Indices</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for (name, ticker), col in zip(INDICES.items(), [c1, c2, c3]):
        p, ch, pct = get_live_data(ticker)
        color = "#00d09c" if ch >= 0 else "#ff4b4b"
        icon = "🟢" if ch >= 0 else "🔴"
        with col:
            st.markdown(f"""
            <div class="fun-card" style="border-top: 3px solid {color};">
                <div style="font-size:12px; color:#888; font-weight:600;">{icon} {name}</div>
                <div style="font-size:24px; font-weight:700; color:#fff;">{p:,.2f}</div>
                <div style="color:{color}; font-weight:600; font-size:14px;">{ch:+.2f} ({pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🛠️ Products & Tools</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    def draw_tool(icon, name, page, col):
        with col:
            if st.button(f"{icon}\n{name}", key=name, use_container_width=True):
                navigate_to(page)
                st.rerun()
    draw_tool("📈", "Stocks", "📈 Stock Analyzer", c1)
    draw_tool("🏦", "ETFs", "🏦 ETFs & Mutual Funds", c2)
    draw_tool("🤖", "AI Picks", "⭐ Top 5 AI Picks", c3)
    draw_tool("🛢️", "Commodities", "🛢️ Global Commodities", c4)

    st.markdown('<div class="section-title">🔥 Most Traded (Live)</div>', unsafe_allow_html=True)
    top_vol_df = scan_most_traded()
    if not top_vol_df.empty:
        cols = st.columns(4)
        for i, (index, row) in enumerate(top_vol_df.iterrows()):
            sym = row['symbol'].replace(".NS","")
            pct = row['pct']
            color_class = "gradient-text-green" if pct >= 0 else "gradient-text-red"
            sign = "+" if pct >= 0 else ""
            with cols[i]:
                st.markdown(f"""
                <div class="fun-card" style="text-align:center;">
                    <img src="https://img.icons8.com/fluency/48/bullish.png" width="35" style="margin-bottom:8px; opacity:0.8;">
                    <div style="font-weight:700; font-size:15px; margin-bottom:4px;">{sym}</div>
                    <div style="font-size:18px; font-weight:bold;">₹{row['price']:.2f}</div>
                    <div class="{color_class}" style="font-size:14px;">{sign}{pct:.2f}%</div>
                    <div style="font-size:10px; color:#666; margin-top:5px;">Vol: {row['vol']/1000000:.1f}M</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📰 Market Updates</div>', unsafe_allow_html=True)
    news = get_news_multi()
    news_cols = st.columns(2)
    for i, n in enumerate(news):
        with news_cols[i % 2]:
            st.markdown(f"""
            <div class="news-box">
                <div class="news-meta">{n['source']} • {n['time']}</div>
                <a href="{n['link']}" target="_blank" class="news-link">{n['title']}</a>
            </div>
            """, unsafe_allow_html=True)

# --------------------------
# 9. OTHER PAGES
else:
    st.markdown(f'<div class="section-title">{view}</div>', unsafe_allow_html=True)
    if st.button("← Back to Dashboard"):
        navigate_to("🏠 Market Dashboard")
        st.rerun()

    if view == "⭐ Top 5 AI Picks":
        try:
            from streamlit_lottie import st_lottie
            lottie_url = "https://assets1.lottiefiles.com/packages/lf20_usmfx6bp.json"
            r = requests.get(lottie_url)
            st_lottie(r.json(), height=180)
        except: pass
        st.warning("⚠️ AI Scan running on limited pool for demo speed...")

    elif view == "📈 Stock Analyzer":
        
        # 1. EXCHANGE & SEARCH
        col_ex, col_search = st.columns([1, 3])
        with col_ex:
            exchange = st.radio("Exchange", ["NSE", "BSE"], horizontal=True)
        
        with col_search:
            # Allows user to type ANY symbol
            search_query = st.text_input("Search Stock (e.g., RELIANCE, ZOMATO, PAYTM)", "RELIANCE")
        
        # Construct Ticker
        suffix = ".NS" if exchange == "NSE" else ".BO"
        clean_query = search_query.upper().strip()
        final_ticker = f"{clean_query}{suffix}"
        
        # 2. FETCH DATA & FIX BLANK CHART ISSUE
        try:
            df = yf.download(final_ticker, period="1y", interval="1d", progress=False)
            
            # FIX: Check if data is empty or MultiIndex columns caused issues
            if df.empty:
                st.error(f"❌ No data found for **{final_ticker}**. Check spelling or try switching Exchange.")
            else:
                # Flat columns if necessary (yfinance update fix)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 3. TABS
                tab1, tab2 = st.tabs(["📊 Technical Chart", "🗣️ Sentiments & Fundamentals"])
                
                with tab1:
                    # Plotly Chart
                    fig = go.Figure(data=[go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close']
                    )])
                    fig.update_layout(
                        template="plotly_dark", 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=0, r=0, t=20, b=0), 
                        height=500,
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.subheader("Social Sentiment (StockTwits)")
                    # StockTwits Widget
                    components.html(f"""
                    <div id="stocktwits-widget-news"></div>
                    <script type="text/javascript" src="https://api.stocktwits.com/addon/widget/2/widget-loader.min.js"></script>
                    <script type="text/javascript">
                    STWT.Widget({{
                        container: 'stocktwits-widget-news', 
                        symbol: '{clean_query}', 
                        width: '100%', 
                        height: '400', 
                        limit: '15', 
                        scrollbars: 'true', 
                        streaming: 'true', 
                        title: '{clean_query} Live Stream', 
                        style: {{
                            link_color: '48515c', 
                            link_hover_color: '48515c', 
                            header_text_color: 'ffffff', 
                            border_color: '333333', 
                            divider_color: '333333', 
                            box_color: '161920', 
                            stream_color: '161920', 
                            text_color: 'ffffff', 
                            time_color: '999999'
                        }}
                    }});
                    </script>
                    """, height=420, scrolling=True)
                    
                    st.markdown("---")
                    st.subheader("Fundamental Deep Dive")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""<a href="https://www.screener.in/company/{clean_query}/" target="_blank" class="ext-btn">📊 View on Screener.in</a>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<a href="https://stockedge.com/search?query={clean_query}" target="_blank" class="ext-btn">📈 View on StockEdge</a>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""<a href="https://www.google.com/finance/quote/{clean_query}:NSE" target="_blank" class="ext-btn">🌏 Google Finance</a>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading data: {e}")

    else:
        st.info("Feature under construction.")
