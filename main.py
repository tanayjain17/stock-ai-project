import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai
import os
import time

# ✅ Setup Gemini API correctly
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# ✅ Tickers to analyze
tickers = ["RELIANCE.NS", "TCS.NS", "SI=F"]


def analyze_market(ticker):
    # A. Get Data
    df = yf.download(ticker, period="1y", interval="1d")

    if df.empty:
        return f"Error: No data found for {ticker}"

    # B. Indicators
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    latest = df.iloc[-1]

    # ✅ AI Score (Danelfin style)
    score = 0
    reasons = []

    if 40 < latest['RSI_14'] < 70:
        score += 3
        reasons.append("Healthy RSI")

    if latest['Close'] > latest['SMA_200']:
        score += 3
        reasons.append("Above 200 SMA")

    if latest['MACDh_12_26_9'] > 0:
        score += 2
        reasons.append("Positive MACD momentum")

    # C. Prompt for Gemini
    prompt = f"""
You are a professional Financial Analyst AI.

Analyze the following technical indicators for {ticker}.

DATA:
- Current Price: {latest['Close']:.2f}
- RSI (14): {latest['RSI_14']:.2f}
- 200 SMA: {latest['SMA_200']:.2f}
- MACD Histogram: {latest['MACDh_12_26_9']:.4f}

AI SCORE: {score}/8
Reasons: {reasons}

TASK:
1. Determine market regime (Bullish/Bearish/Neutral).
2. Suggest Buy, Sell, or Wait.
3. If Silver (SI=F), consider high volatility.

OUTPUT:
- Verdict
- Confidence
- Reasoning
"""

    response = model.generate_content(prompt)
    return response.text


# ✅ Run for all tickers
if __name__ == "__main__":
    print("--- STARTING AI ANALYSIS ---")

    for t in tickers:
        try:
            print(f"\nAnalyzing {t}...\n")
            result = analyze_market(t)
            print(result)
            time.sleep(4)
        except Exception as e:
            print(f"Failed to analyze {t}: {e}")
