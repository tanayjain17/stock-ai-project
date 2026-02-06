import gdown
from tensorflow.keras.models import load_model
import joblib
import os

# Only download if not already present
if not os.path.exists("stock_model.h5"):
    MODEL_ID = "YOUR_MODEL_FILE_ID"
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", "stock_model.h5", quiet=False)

if not os.path.exists("stock_scaler.gz"):
    SCALER_ID = "YOUR_SCALER_FILE_ID"
    gdown.download(f"https://drive.google.com/uc?id={SCALER_ID}", "stock_scaler.gz", quiet=False)

# Load model and scaler
pretrained_model = load_model("stock_model.h5")
pretrained_scaler = joblib.load("stock_scaler.gz")
