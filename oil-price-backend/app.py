# src/app.py
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from React

# Load data (e.g., historical Brent oil prices and analysis results)
data = pd.read_csv("../data/event-data.csv")

# Endpoint to get oil price data
@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(data.to_dict(orient='records'))

# Endpoint for model metrics (RMSE, MAE)
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics = {
        "rmse": 2.34,  # Example values, calculate these in your analysis
        "mae": 1.78
    }
    return jsonify(metrics)

# Endpoint for events data
@app.route('/api/events', methods=['GET'])
def get_events():
    events = [
            

        {"date": "1990-03-15", "description": "Global economic recession starts, impacting oil prices."},
        {"date": "2001-09-11", "description": "9/11 terrorist attacks cause significant market volatility."},
        {"date": "2008-06-29", "description": "Global financial crisis worsens, leading to a drop in oil prices."},
        {"date": "2010-04-20", "description": "BP oil spill in the Gulf of Mexico causes oil prices to fluctuate."},
        {"date": "2014-11-27", "description": "OPEC announces decision to maintain oil production, causing oil prices to drop."},
        {"date": "2016-01-16", "description": "Iran nuclear deal leads to the lifting of sanctions, impacting oil prices."},
        {"date": "2020-03-11", "description": "COVID-19 pandemic triggers a sharp decline in global oil demand."},
        {"date": "2022-02-24", "description": "Russia invades Ukraine, causing a spike in oil prices due to supply disruptions."}
    ]
    return jsonify(events)

if __name__ == '__main__':
    app.run(debug=True)
