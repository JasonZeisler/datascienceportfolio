import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('btc_price_data.csv')

prices = data['close'].values
prices = prices.reshape(-1, 1)

# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Prepare the data for RNN input
def prepare_data(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence) - 1:
            break
        X.append(sequence[i:end_idx])
        y.append(sequence[end_idx])
    return np.array(X), np.array(y)

n_steps = 10  # Number of past days to consider
X, y = prepare_data(scaled_prices, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(n_steps, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Predictions
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the results
plt.plot(prices, color='blue', label='Original Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')

# plot moving averages
plt.plot(data['ma_50'], color='green', label='50-day SMA')
plt.plot(data['ma_200'], color='purple', label='200-day SMA')

plt.title('Price Prediction')
plt.legend()
plt.show()
