import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import os
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# ✅ Homepage Route (Avoids 404 Errors)
@app.route("/")
def home():
    return jsonify({"message": "Stock Prediction API is running!"})

# ✅ Function to Fetch Live Stock Data
def get_stock_data(symbol):
    try:
        df = yf.download(symbol, period="70d")
        if df.empty:
            return None
        return df
    except Exception as e:
        return None

# ✅ Function to Compute Technical Indicators
def compute_indicators(df):
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df.ffill().bfill()

# ✅ Function to Fetch News Sentiment
def get_news_sentiment(symbol):
    try:
        url = f"https://www.bing.com/news/search?q={symbol}+stock"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = [h.text for h in soup.find_all("a") if h.text.strip()]
        sentiment_score = np.mean([TextBlob(h).sentiment.polarity for h in headlines]) if headlines else 0
        return sentiment_score
    except Exception as e:
        return 0  # Default to neutral sentiment

# ✅ Prediction Route
@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol", "").upper()
    
    if not symbol:
        return jsonify({"error": "Stock symbol is required"}), 400
    
    df = get_stock_data(symbol)
    if df is None:
        return jsonify({"error": "Invalid stock symbol or no data available"}), 400

    df = compute_indicators(df)
    df["Sentiment"] = get_news_sentiment(symbol)

    features = ["Close", "SMA_50", "EMA_50", "RSI", "MACD", "Signal", "Sentiment"]

    model_path = f"models/{symbol}_lstm_model.h5"
    scaler_X_path = f"models/{symbol}_scaler_X.pkl"
    scaler_y_path = f"models/{symbol}_scaler_y.pkl"

    # ✅ Check if LSTM Model Exists
    if os.path.exists(model_path):
        model = load_model(model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)

        scaled_features = scaler_X.transform(df[features])
        sequence = scaled_features[-60:].reshape(1, 60, len(features))

        # Predict with LSTM Model
        lstm_pred_scaled = model.predict(sequence)
        lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1))[0, 0]
    
    else:
        # ✅ Use XGBoost as Fallback
        xgb_path = f"models/{symbol}_xgb.pkl"
        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            scaled_features = joblib.load(f"models/{symbol}_scaler_X.pkl").transform(df[features])
            lstm_pred = joblib.load(f"models/{symbol}_scaler_y.pkl").inverse_transform(
                xgb_model.predict(scaled_features[-1].reshape(1, -1)).reshape(-1, 1)
            )[0, 0]
        else:
            return jsonify({"error": "Model not found for this stock. Please train first!"}), 404

    # ✅ Response
    return jsonify({
        "symbol": symbol,
        "current_price": round(df["Close"].iloc[-1], 2),
        "predicted_price": round(lstm_pred, 2),
        "technical_indicators": {
            "SMA_50": round(df["SMA_50"].iloc[-1], 2),
            "EMA_50": round(df["EMA_50"].iloc[-1], 2),
            "RSI": round(df["RSI"].iloc[-1], 2),
            "MACD": round(df["MACD"].iloc[-1], 2),
            "Signal_Line": round(df["Signal"].iloc[-1], 2),
        },
        "sentiment_score": round(df["Sentiment"].iloc[-1], 4)
    })

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
