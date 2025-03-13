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
@app.route("/predict", methods=["GET
