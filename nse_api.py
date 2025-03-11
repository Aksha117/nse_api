from flask import Flask, jsonify
from nsetools import Nse
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access
nse = Nse()

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
    return 100 - (100 / (1 + rs))

# Function to calculate SMA (Simple Moving Average)
def calculate_sma(prices, period=50):
    if len(prices) < period:
        return None  # Not enough data
    return np.mean(prices[-period:])

# Function to calculate EMA (Exponential Moving Average)
def calculate_ema(prices, period=50):
    if len(prices) < period:
        return None
    return np.round(np.average(prices[-period:], weights=np.exp(np.linspace(-1., 0., period))), 2)

@app.route('/get_stock/<symbol>', methods=['GET'])
def get_stock(symbol):
    stock_data = nse.get_quote(symbol.upper())  # Fetch stock data
    
    if stock_data:  # Ensure stock data exists
        last_price = stock_data.get('lastPrice', 0)
        previous_closes = [stock_data.get('previousClose', 0)] * 50  # Dummy data (Replace with historical prices)

        rsi = calculate_rsi(previous_closes)
        sma_50 = calculate_sma(previous_closes, 50)
        ema_50 = calculate_ema(previous_closes, 50)

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
