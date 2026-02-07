# ========================= train_model.py =========================
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from xgboost import XGBClassifier

print("Downloading data...")

symbol = "RELIANCE.NS"
end = datetime.now()
start = end - timedelta(days=365*3)

df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True)
df.dropna(inplace=True)

# ================= FEATURE ENGINEERING (quant grade) =================
df['Returns'] = df['Close'].pct_change()
df['SMA20'] = df['Close'].rolling(20).mean()
df['EMA20'] = EMAIndicator(df['Close'], 20).ema_indicator()

df['RSI'] = RSIIndicator(df['Close']).rsi()

macd = MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()

bb = BollingerBands(df['Close'])
df['BB_H'] = bb.bollinger_hband()
df['BB_L'] = bb.bollinger_lband()
df['BB_W'] = df['BB_H'] - df['BB_L']

df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()

df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

# Target: next 3-day direction
df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)

df.dropna(inplace=True)

features = [
    'Returns','SMA20','EMA20','RSI','MACD','MACD_Signal',
    'BB_H','BB_L','BB_W','ATR','ADX','OBV'
]

X = df[features]
y = df['Target']

print("Training XGBoost...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X, y)

joblib.dump((model, features), "model.pkl")
print("✅ model.pkl saved!")
