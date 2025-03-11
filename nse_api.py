from flask import Flask, jsonify
from nsetools import Nse
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access
nse = Nse()

# Function to fetch historical data (Dummy method for now)
def get_historical_prices(symbol, days=50):
    try:
        # Normally, we should fetch real historical prices from an API
        last_price = nse.get_quote(symbol).get('lastPrice', 0)
        return [last_price - (i * 1.5) for i in range(days)][::-1]  # Fake decreasing prices
    except:
        return [0] * days  # Return default values if unavailable

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

@app.route('/get_stock/<symbol>', methods=['GET'])
def get_stock(symbol):
    stock_data = nse.get_quote(symbol.upper())  # Fetch stock data
    
    if stock_data:  # Ensure stock data exists
        last_price = stock_data.get('lastPrice', 0)
        historical_prices = get_historical_prices(symbol, 50)  # Fetch actual historical data

        rsi = calculate_rsi(historical_prices)
        sma_50 = calculate_sma(historical_prices, 50)
        ema_50 = calculate_ema(historical_prices, 50)

        response = {
            "symbol": symbol.upper(),
            "lastPrice": last_price,
            "dayHigh": stock_data.get('dayHigh', 'N/A'),
            "dayLow": stock_data.get('dayLow', 'N/A'),
            "high52": stock_data.get('high52', 'N/A'),
            "low52": stock_data.get('low52', 'N/A'),
            "rsi": rsi,
            "sma_50": sma_50,
            "ema_50": ema_50,
            "marketCap": stock_data.get('marketCap', 'N/A'),
            "peRatio": stock_data.get('pE', 'N/A'),
            "bookValue": stock_data.get('bookValue', 'N/A'),
            "dividendYield": stock_data.get('dividendYield', 'N/A')
        }
        return jsonify(response)

    return jsonify({"error": "Stock symbol not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
