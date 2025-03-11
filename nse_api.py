from flask import Flask, jsonify
from nsetools import Nse
from flask_cors import CORS
import numpy as np
import yfinance as yf  # Import Yahoo Finance

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access
nse = Nse()

# Function to fetch real historical stock prices from Yahoo Finance
def get_historical_prices(symbol, days=50):
    try:
        yahoo_symbol = f"{symbol}.NS"  # Convert NSE symbol to Yahoo format
        stock = yf.Ticker(yahoo_symbol)
        history = stock.history(period=f"{days}d")
        return list(history['Close'].values)  # Return closing prices
    except:
        return [0] * days  # Default values if data is unavailable

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(prices, period=14):
    prices = np.array(prices)
    delta = np.diff(prices)

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    for i in range(period, len(prices) - 1):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

# Function to calculate SMA (Simple Moving Average)
def calculate_sma(prices, period=50):
    if len(prices) < period:
        return None  # Not enough data
    return round(np.mean(prices[-period:]), 2)

# Function to calculate EMA (Exponential Moving Average)
def calculate_ema(prices, period=50):
    if len(prices) < period:
        return None
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
    stock_data = nse.get_quote(symbol.upper())  # Fetch stock data from NSE
    
    if stock_data:  # Ensure stock data exists
        last_price = stock_data.get('lastPrice', 0)
        historical_prices = get_historical_prices(symbol, 50)  # Fetch actual historical data

        # Calculate technical indicators
        rsi = calculate_rsi(historical_prices)
        sma_50 = calculate_sma(historical_prices, 50)
        ema_50 = calculate_ema(historical_prices, 50)
        upper_band, lower_band = calculate_bollinger_bands(historical_prices, 20)

        # Fetch fundamental indicators properly
        response = {
            "symbol": symbol.upper(),
            "lastPrice": last_price,
            "dayHigh": stock_data.get('dayHigh', stock_data.get('intraDayHighLow', {}).get('max', 'N/A')),
            "dayLow": stock_data.get('dayLow', stock_data.get('intraDayHighLow', {}).get('min', 'N/A')),
            "high52": stock_data.get('high52', stock_data.get('weekHighLow', {}).get('max', 'N/A')),
            "low52": stock_data.get('low52', stock_data.get('weekHighLow', {}).get('min', 'N/A')),
            "rsi": rsi,
            "sma_50": sma_50,
            "ema_50": ema_50,
            "bollinger_upper": upper_band,
            "bollinger_lower": lower_band,
            "marketCap": stock_data.get('marketCap', stock_data.get('info', {}).get('marketCap', 'N/A')),
            "peRatio": stock_data.get('pE', stock_data.get('info', {}).get('pE', 'N/A')),
            "bookValue": stock_data.get('bookValue', stock_data.get('info', {}).get('bookValue', 'N/A')),
            "dividendYield": stock_data.get('dividendYield', stock_data.get('info', {}).get('dividendYield', 'N/A'))
        }
        return jsonify(response)

    return jsonify({"error": "Stock symbol not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
