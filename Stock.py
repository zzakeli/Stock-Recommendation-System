import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 2])  # Predicting Close price
    return np.array(X), np.array(y)

# Define stock symbols
stock_symbols = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']

# Define date range for historical data
start_date = '2020-01-01'
end_date = '2025-03-29'

seq_length = 60  # Define sequence length
predictions = {}

for symbol in stock_symbols:
    # Fetch historical stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    # Extract VOCHL features (Volume, Open, Close, High, Low)
    stock_features = stock_data[['Volume', 'Open', 'Close', 'High', 'Low']].values
    
    # Split data into training (80%) and testing (20%)
    train_size = int(len(stock_features) * 0.8)
    train_data, test_data = stock_features[:train_size], stock_features[train_size:]
    
    # Scale data using only training data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    # Define CNN + LSTM model with L2 regularization
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', 
               kernel_regularizer=l2(0.0009), input_shape=(seq_length, 5)),  # L2 regularization
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False, 
             kernel_regularizer=l2(0.0009)),  # L2 regularization
        Dropout(0.2),
        Dense(1, kernel_regularizer=l2(0.0009))  # L2 regularization
    ])
    
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    
    # Evaluate model on test data
    train_loss = history.history['loss'][-1]  # Get last training loss
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test Loss for {symbol}: {test_loss:.6f}")
    print(f"MAE for {symbol}: {mae:.6f}")
    print(f"RMSE for {symbol}: {rmse:.6f}")
    print(f"Training Loss for {symbol}: {train_loss:.6f}")

    # Predict the next day's price using the last available sequence
    last_sequence = np.array(test_scaled[-seq_length:]).reshape(1, seq_length, 5)
    predicted_price_scaled = model.predict(last_sequence)
    
    # Inverse transform only the Close price (index 2)
    placeholder = np.zeros((1, 5))  # Placeholder array with correct shape
    placeholder[0, 2] = predicted_price_scaled[0, 0]  # Insert predicted Close price
    predicted_price = scaler.inverse_transform(placeholder)[0, 2]  # Extract Close price
    predictions[symbol] = predicted_price

# Rank stocks based on predictions
ranked_stocks = pd.DataFrame(list(predictions.items()), columns=['Stock', 'Predicted Price'])
ranked_stocks = ranked_stocks.sort_values(by='Predicted Price', ascending=False).reset_index(drop=True)
print(ranked_stocks.head(7))
