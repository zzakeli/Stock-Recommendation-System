import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Embedding, Dropout, Conv1D, LSTM, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def compute_rsi(data, window=14):
    """Computes the Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_sequences(data, stock_id, seq_length):
    """Creates sequences for LSTM, including stock identifier"""
    X, y, stock_labels = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 2])  # Predict Close price
        stock_labels.append(stock_id)  # Append stock identifier
    return np.array(X), np.array(y), np.array(stock_labels)

# Define stock symbols
stock_symbols = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
start_date, end_date = '2020-01-01', '2025-03-29'
seq_length = 60

# Store combined data
global_scaler = MinMaxScaler()
all_sequences, all_labels, all_stock_ids = [], [], []

# Load data for all stocks
stock_mapping = {symbol: idx for idx, symbol in enumerate(stock_symbols)}

global_data = None
stock_stats = {}
for symbol in stock_symbols:
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate SMA, EMA, and RSI
    stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
    stock_data['RSI_14'] = compute_rsi(stock_data)
    stock_data['Returns'] = stock_data['Close'].pct_change()

    # Drop NaN values
    stock_data.dropna(inplace=True)
    
    stock_stats[symbol] = {
        'mean_return': stock_data['Returns'].mean(),
        'std_dev': stock_data['Returns'].std(),
        'rsi': stock_data['RSI_14'].iloc[-1]
    }

    # Select features
    stock_features = stock_data[['Volume', 'Open', 'Close', 'High', 'Low', 'SMA_10', 'EMA_10', 'RSI_14']].values

    # Store scaled data globally
    global_data = stock_features if global_data is None else np.vstack((global_data, stock_features))

# Fit global scaler on all stock data
global_scaler.fit(global_data)

# Process each stock again for training
for symbol in stock_symbols:
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate SMA, EMA, and RSI
    stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
    stock_data['RSI_14'] = compute_rsi(stock_data)

    # Drop NaN values
    stock_data.dropna(inplace=True)

    # Select features
    stock_features = stock_data[['Volume', 'Open', 'Close', 'High', 'Low', 'SMA_10', 'EMA_10', 'RSI_14']].values
    
    # Normalize using the global scaler
    stock_features_scaled = global_scaler.transform(stock_features)
    
    # Create sequences
    X, y, stock_ids = create_sequences(stock_features_scaled, stock_mapping[symbol], seq_length)
    all_sequences.append(X)
    all_labels.append(y)
    all_stock_ids.append(stock_ids)

# Convert lists to numpy arrays
X = np.vstack(all_sequences)
y = np.concatenate(all_labels)
stock_ids = np.concatenate(all_stock_ids)

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test, stock_ids_train, stock_ids_test = train_test_split(
    X, y, stock_ids, test_size=0.2, random_state=42
)

# Compute sample weights (more weight to stable stocks)
# volatility = {symbol: np.std(y_train[stock_ids_train == stock_mapping[symbol]]) for symbol in stock_symbols}
# max_volatility = max(volatility.values())
# sample_weights = np.array([max_volatility / volatility[stock_symbols[stock_id]] for stock_id in stock_ids_train])

# Define model with stock embedding
input_data = Input(shape=(seq_length, 8))  # Updated shape to include RSI
stock_input = Input(shape=(1,))  # Stock ID input

stock_embedding = Embedding(len(stock_symbols), 3)(stock_input)
stock_embedding = Flatten()(stock_embedding)

cnn = Conv1D(64, 2, activation='relu', kernel_regularizer=l2(0.0009))(input_data)
cnn = Dropout(0.2)(cnn)

lstm = LSTM(50, activation='relu', return_sequences=False, kernel_regularizer=l2(0.0009))(cnn)
lstm = Dropout(0.2)(lstm)

merged = Concatenate()([lstm, stock_embedding])
output = Dense(1, kernel_regularizer=l2(0.0009))(merged)

model = Model(inputs=[input_data, stock_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Train model with sample weights and track history
history = model.fit(
    [X_train, stock_ids_train], y_train, 
    epochs=50, batch_size=32, verbose=1, 
    validation_data=([X_test, stock_ids_test], y_test)  # Track validation loss
)

# Evaluate test loss
test_loss = model.evaluate([X_test, stock_ids_test], y_test)
print(f"Test Loss: {test_loss:.4f}")

# Plot training vs. validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training vs. Validation Loss')
# plt.show()

# Make predictions
predictions = model.predict([X_test, stock_ids_test])
predictions = global_scaler.inverse_transform(np.column_stack([np.zeros((predictions.shape[0], 7)), predictions]))[:, -1]
y_test = global_scaler.inverse_transform(np.column_stack([np.zeros((y_test.shape[0], 7)), y_test]))[:, -1]

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse:.4f}")

# Plot actual vs. predicted stock prices
# plt.figure(figsize=(10, 5))
# plt.plot(y_test, label="Actual Prices", color="blue")
# plt.plot(predictions, label="Predicted Prices", color="orange")
# plt.legend()
# plt.title("Actual vs. Predicted Stock Prices")
# plt.xlabel("Test Sample")
# plt.ylabel("Scaled Price")
# plt.show()

investment_scores = {}
for symbol in stock_symbols:
    avg_pred = np.mean(predictions[np.where(stock_ids_test == stock_mapping[symbol])])
    mean_return = stock_stats[symbol]['mean_return']
    risk = stock_stats[symbol]['std_dev'] + 1e-6  # Avoid division by zero                      
    rsi = stock_stats[symbol]['rsi']
    rsi_penalty = 1 - abs((rsi - 50) / 100)  # Smoother RSI adjustment
    investment_score = (mean_return / risk) * avg_pred * rsi_penalty  
    investment_scores[symbol] = investment_score

ranked_stocks = sorted(investment_scores.items(), key=lambda x: x[1], reverse=True)

print("Stock Rankings (Based on Investment Potential):")
for rank, (stock, score) in enumerate(ranked_stocks, 1):
    print(f"{rank}. {stock} - Investment Score: {score:.4f}")

model.save("Stock_Recommendation_Model.h5")
