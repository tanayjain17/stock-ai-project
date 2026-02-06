import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai
import os
import time

# 1. SETUP: Configure the AI with your Key
# We look for the key in the environment variables for security
genai.configure(api_key=os.environ)
model = genai.GenerativeModel('gemini-2.0-flash')

# 2. TARGETS: Silver Futures (SI=F) and Stocks
tickers = 

def analyze_market(ticker):
    # A. GET DATA: Download last 1 year of data from Yahoo Finance
    # We use 'SI=F' for Silver Futures
    df = yf.download(ticker, period="1y", interval="1d")
    
    if df.empty:
        return f"Error: No data found for {ticker}"

    # B. CALCULATE INDICATORS (The Math)
    # RSI (Momentum): Is it overbought?
    df = df.ta.rsi(length=14)
    # SMA (Trend): Is the price above the 200-day average?
    df = df.ta.sma(length=200)
    # MACD (Momentum strength)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = df.join(macd)

    # Get the latest row of data
    latest = df.iloc[-1]

    # C. THE "PROPER PROMPT" 
    # This is the strict instruction we send to the AI
    prompt = f"""
    You are a professional Financial Analyst AI. 
    Analyze the following technical indicators for {ticker} as of today.
    
    DATA CONTEXT:
    - Current Price: {latest['Close']:.2f}
    - RSI (14-day): {latest:.2f} (Over 70=Overbought, Under 30=Oversold)
    - 200-Day SMA: {latest:.2f} (Price above this = Long Term Bullish)
    - MACD Histogram: {latest:.4f} (Positive = Upward Momentum)

    TASK:
    1. Determine the Market Regime (Bullish/Bearish/Neutral).
    2. Suggest a specific strategy (Buy, Sell, or Wait).
    3. If this is Silver (SI=F), account for high volatility and "fake-out" moves.

    OUTPUT FORMAT:
    - Verdict:
    - Confidence: [0-100%]
    - Reasoning: [One concise paragraph]
    """

    # D. SEND TO CLOUD BRAIN
    response = model.generate_content(prompt)
    return response.text

# 3. RUN THE ANALYSIS
if __name__ == "__main__":
    print("--- STARTING AI ANALYSIS ---")
    for t in tickers:
        try:
            print(f"\nAnalyzing {t}...")
            result = analyze_market(t)
            print(result)
            # Sleep 4 seconds to respect the Free Tier rate limits (15 requests/min)
            time.sleep(4) 
        except Exception as e:
            print(f"Failed to analyze {t}: {e}")