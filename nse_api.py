from flask import Flask, jsonify
from nsetools import Nse
from flask_cors import CORS  # Allow frontend to call this API

app = Flask(__name__)
CORS(app)  # Enable CORS for CodePen requests
nse = Nse()

@app.route('/get_stock/<symbol>', methods=['GET'])
def get_stock(symbol):
    stock_data = nse.get_quote(symbol.upper())  # Fetch real-time data from NSE
    
    if stock_data:  # Ensure stock data exists
        response = {
            "symbol": symbol.upper(),
            "lastPrice": stock_data.get('lastPrice', 'N/A'),
            "dayHigh": stock_data.get('intraDayHighLow', {}).get('max', 'N/A'),
            "dayLow": stock_data.get('intraDayHighLow', {}).get('min', 'N/A'),
            "high52": stock_data.get('weekHighLow', {}).get('max', 'N/A'),
            "low52": stock_data.get('weekHighLow', {}).get('min', 'N/A')
        }
        return jsonify(response)
    
    return jsonify({"error": "Stock symbol not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
