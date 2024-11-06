import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

def analyze_brent_prices_with_arima(brent_data, forecast_steps=5):
    # Ensure 'Date' column is in datetime format and set as index
    if 'Date' in brent_data.columns:
        brent_data['Date'] = pd.to_datetime(brent_data['Date'])
        brent_data.set_index('Date', inplace=True)

    # Check for stationarity
    result = adfuller(brent_data['Price'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    # If not stationary, difference the series
    if result[1] > 0.05:  # p-value greater than 0.05 indicates non-stationarity
        brent_data['Price_diff'] = brent_data['Price'].diff().dropna()
        model_data = brent_data['Price_diff'].dropna()
    else:
        model_data = brent_data['Price']

    # Fit the ARIMA model (adjust p, d, q as needed)
    model = ARIMA(model_data, order=(1, 1, 1))  # Set appropriate orders based on your analysis
    model_fit = model.fit()

    # Forecasting
    forecast = model_fit.forecast(steps=forecast_steps)

    # Plotting the historical prices and forecast
    plt.figure(figsize=(12, 6))
    plt.plot(brent_data.index, brent_data['Price'], label='Brent Oil Price', color='blue')

    # Generate the forecast index
    forecast_index = pd.date_range(start=brent_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)

    plt.plot(forecast_index, forecast, label='Forecast', color='red')
    plt.title('Brent Oil Price Forecast using ARIMA Model')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per barrel)')
    plt.legend()
    plt.show()

    return forecast, model_fit  # Return forecasted values and fitted model


def analyze_brent_prices_with_ms_arima(brent_data, forecast_steps=5):
    # Ensure 'Date' column is in datetime format and set as index
    if 'Date' in brent_data.columns:
        brent_data['Date'] = pd.to_datetime(brent_data['Date'])
        brent_data.set_index('Date', inplace=True)

    # Check for stationarity
    result = adfuller(brent_data['Price'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    # If the series is not stationary, difference the series
    if result[1] > 0.05:  # p-value greater than 0.05 indicates non-stationarity
        brent_data['Price_diff'] = brent_data['Price'].diff().dropna()
        model_data = brent_data['Price_diff'].dropna()
    else:
        model_data = brent_data['Price']

    # Fit the Markov-Switching Regression model
    ms_model = MarkovRegression(model_data, k_regimes=2, trend='c', switching_variance=True)
    ms_model_fit = ms_model.fit()

    # Summary of the model
    print(ms_model_fit.summary())

    # Forecasting
    forecast = ms_model_fit.forecast(steps=forecast_steps)

    # Plotting the historical prices and forecast
    plt.figure(figsize=(12, 6))
    plt.plot(brent_data.index, brent_data['Price'], label='Brent Oil Price', color='blue')

    # Generate the forecast index
    forecast_index = pd.date_range(start=brent_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)

    plt.plot(forecast_index, forecast, label='Forecast', color='red')
    plt.title('Brent Oil Price Forecast using Markov-Switching ARIMA Model')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per barrel)')
    plt.legend()
    plt.show()

    return forecast, ms_model_fit

def prepare_data_for_lstm(brent_file_path, time_steps=1):
    # Load the Brent oil prices data
    brent_data = pd.read_csv(brent_file_path)
    
    # Convert 'Date' column to datetime and set as index
    brent_data['Date'] = pd.to_datetime(brent_data['Date'])
    brent_data.set_index('Date', inplace=True)
    
    # Scale the data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(brent_data['Price'].values.reshape(-1, 1))
    
    # Prepare the data for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:(i + time_steps), 0])
        y.append(scaled_data[i + time_steps, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be [samples, time steps, features] for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def create_and_train_lstm(X_train, y_train, epochs=50, batch_size=32):
    # Build the LSTM model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.LSTM(50, return_sequences=False))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))  # Output layer
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    return model
def forecast_with_lstm(model, scaler, brent_data, time_steps=1, forecast_steps=5):
    # Create the input for forecasting
    last_data = brent_data['Price'].values[-time_steps:]
    last_data = last_data.reshape(-1, 1)
    last_scaled = scaler.transform(last_data)

    X_forecast = []
    for _ in range(forecast_steps):
        X_forecast.append(last_scaled)
        last_scaled = np.append(last_scaled[1:], [[last_scaled[-1]]]).reshape(-1, 1)
    
    X_forecast = np.array(X_forecast)
    X_forecast = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], 1))
    
    # Make predictions
    predictions = model.predict(X_forecast)
    predictions = scaler.inverse_transform(predictions)  # Reverse scaling

    return predictions
def plot_lstm_predictions(brent_data, predictions, scaler, time_steps=30):
    # Inverse transform the predictions to original scale
    predictions_inverse = scaler.inverse_transform(predictions)

    # Prepare historical data for plotting
    train_data = brent_data['Price'].values
    train_data_inverse = scaler.inverse_transform(train_data.reshape(-1, 1))

    # Create an array for the total length of the time series
    full_data = np.empty((len(train_data) + len(predictions),))
    full_data[:len(train_data)] = train_data_inverse[:, 0]
    full_data[len(train_data):] = np.nan  # Placeholder for predictions

    # Plot the historical data
    plt.figure(figsize=(12, 6))
    plt.plot(brent_data.index, train_data_inverse, label='Brent Oil Price', color='blue')

    # Create forecast index
    forecast_index = pd.date_range(start=brent_data.index[-1] + pd.Timedelta(days=1), 
                                    periods=len(predictions))

    # Plot the predictions
    plt.plot(forecast_index, predictions_inverse, label='LSTM Forecast', color='red')

    plt.title('Brent Oil Price Forecast using LSTM Model')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per barrel)')
    plt.legend()
    plt.grid()
    plt.show()

# Define helper function to calculate metrics
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Backtesting function
def backtest_arima(prices, order=(1, 1, 1), train_size=0.8):
    train_size = int(len(prices) * train_size)
    train, test = prices[:train_size], prices[train_size:]
    
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=len(test))
    metrics = calculate_metrics(test, forecast)
    
    print("Backtest Metrics:", metrics)
    
    plt.figure(figsize=(12, 6))
    plt.plot(train, label="Train")
    plt.plot(test, label="Test", color="orange")
    plt.plot(test.index, forecast, label="Forecast", color="green")
    plt.legend()
    plt.show()

    return metrics

# Out-of-sample testing function
def out_of_sample_testing(prices, order=(1, 1, 1), out_of_sample_period=30):
    train, test = prices[:-out_of_sample_period], prices[-out_of_sample_period:]
    
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=out_of_sample_period)
    metrics = calculate_metrics(test, forecast)
    
    print("Out-of-Sample Metrics:", metrics)
    
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Full Data")
    plt.plot(test.index, forecast, label="Out-of-Sample Forecast", color="red")
    plt.legend()
    plt.show()

    return metrics

# Time series cross-validation function
def time_series_cross_validation(prices, order=(1, 1, 1), n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []

    for train_index, test_index in tscv.split(prices):
        train, test = prices[train_index], prices[test_index]
        
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=len(test))
        metrics = calculate_metrics(test, forecast)
        errors.append(metrics)
        print(f"Cross-Validation Fold Metrics: {metrics}")

    avg_metrics = {metric: np.mean([fold[metric] for fold in errors]) for metric in errors[0]}
    print("Average Cross-Validation Metrics:", avg_metrics)

    return avg_metrics
