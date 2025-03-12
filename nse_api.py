from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import yfinance as yf
from nsetools import Nse

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access
nse = Nse()

# Function to fetch historical stock prices from Yahoo Finance
def get_historical_prices(symbol, days=50):
    try:
        yahoo_symbol = f"{symbol}.NS"  # Convert NSE symbol to Yahoo format
        stock = yf.Ticker(yahoo_symbol)
        history = stock.history(period=f"{days}d")

        if history.empty:
            return [0] * days  # Return default values if unavailable

        return list(history['Close'].values)  # Return closing prices

    except Exception as e:
        print(f"Error fetching historical prices from Yahoo: {e}")
        return [0] * days  # Default values if data is unavailable

# Function to fetch real-time stock price from NSE
def get_nse_stock_price(symbol):
    try:
        stock_data = nse.get_quote(symbol.upper())
        return stock_data.get('lastPrice', None)
    except Exception as e:
        print(f"Error fetching price from NSE: {e}")
        return None  # Return None if NSE data is unavailable

# Function to fetch stock price from Yahoo Finance
def get_yahoo_stock_price(symbol):
    try:
        yahoo_symbol = f"{symbol}.NS"
        stock = yf.Ticker(yahoo_symbol)
        data = stock.history(period="1d")
        
        if not data.empty:
            return data["Close"].iloc[-1]  # Last closing price

    except Exception as e:
        print(f"Error fetching price from Yahoo: {e}")

    return None  # Return None if Yahoo data is unavailable

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return None  # Not enough data
    
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    for i in range(period, len(prices) - 1):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period

    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)

# Function to calculate SMA (Simple Moving Average)
def calculate_sma(prices, period=50):
    if len(prices) < period:
        return None  # Not enough data
    return round(np.mean(prices[-period:]), 2)

# Function to calculate EMA (Exponential Moving Average)
def calculate_ema(prices, period=50):
    if len(prices) < period:
        return None  # Not enough data
    return round(np.average(prices[-period:], weights=np.exp(np.linspace(-1., 0., period))), 2)

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(prices, period=20, std_dev=2):
    if len(prices) < period:
        return None, None  # Not enough data
    
    sma = calculate_sma(prices, period)
    std = np.std(prices[-period:])
    
    upper_band = round(sma + (std_dev * std), 2)
    lower_band = round(sma - (std_dev * std), 2)
    
    return upper_band, lower_band

# Function to fetch stock data from NSE & Yahoo Finance
@app.route('/get_stock/<symbol>', methods=['GET'])
def get_stock(symbol):
    try:
        yahoo_symbol = f"{symbol}.NS"
        stock = yf.Ticker(yahoo_symbol)
        stock_data = stock.history(period="1d")

        # Fetch data from NSE (if available)
        nse_price = get_nse_stock_price(symbol)

        # Fetch data from Yahoo (as a backup)
        yahoo_price = get_yahoo_stock_price(symbol)

        # Decide which price to use (Prefer NSE, fallback to Yahoo)
        last_price = nse_price if nse_price is not None else yahoo_price

        if last_price is None:
            return jsonify({"error": "Stock data not found"}), 404

        # Fetch historical prices from Yahoo for technical indicators
        historical_prices = get_historical_prices(symbol, 50)

        # Calculate technical indicators
        rsi = calculate_rsi(historical_prices)
        sma_50 = calculate_sma(historical_prices, 50)
        ema_50 = calculate_ema(historical_prices, 50)
        upper_band, lower_band = calculate_bollinger_bands(historical_prices, 20)

        # Fetch fundamental indicators
        stock_info = stock.info
        response = {
            "symbol": symbol.upper(),
            "lastPrice": last_price,
            "nsePrice": nse_price,
            "yahooPrice": yahoo_price,
            "dayHigh": stock_data["High"].iloc[-1] if not stock_data.empty else "N/A",
            "dayLow": stock_data["Low"].iloc[-1] if not stock_data.empty else "N/A",
            "high52": stock_info.get("fiftyTwoWeekHigh", "N/A"),
            "low52": stock_info.get("fiftyTwoWeekLow", "N/A"),
            "rsi": rsi,
            "sma_50": sma_50,
            "ema_50": ema_50,
            "bollinger_upper": upper_band,
            "bollinger_lower": lower_band,
            "marketCap": stock_info.get("marketCap", "N/A"),
            "peRatio": stock_info.get("trailingPE", "N/A"),
            "bookValue": stock_info.get("bookValue", "N/A"),
            "dividendYield": stock_info.get("dividendYield", "N/A")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Failed to fetch stock data: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
