import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, jsonify
from tensorflow.keras.layers import Input, Embedding, Dropout, Conv1D, LSTM, Dense, Flatten, Concatenate # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from flask import render_template
import os

app = Flask(__name__)

# Define stock symbols
stock_symbols = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
start_date, end_date = '2020-01-01', '2025-03-29'
seq_length = 60

global_scaler = MinMaxScaler()
stock_mapping = {symbol: idx for idx, symbol in enumerate(stock_symbols)}

# ------------------------- FUNCTIONS -------------------------
def compute_rsi(data, window=14):
    """Computes the Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_sequences(data, stock_id, seq_length):
    """Creates sequences for LSTM"""
    X, y, stock_labels = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 2])  # Predict Close price
        stock_labels.append(stock_id)
    return np.array(X), np.array(y), np.array(stock_labels)

stock_stats = {}
def load_stock_data():
    """Loads and processes stock data"""
    global global_scaler
    all_sequences, all_labels, all_stock_ids = [], [], []
    
    global_data = None
    for symbol in stock_symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"⚠ Warning: No data found for {symbol}")
            continue

        stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
        stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
        stock_data['RSI_14'] = compute_rsi(stock_data)
        stock_data['Returns'] = stock_data['Close'].pct_change()
        
        stock_data.dropna(inplace=True)
        
        stock_stats[symbol] = {
        'mean_return': stock_data['Returns'].mean(),
        'std_dev': stock_data['Returns'].std(),
        'rsi': stock_data['RSI_14'].iloc[-1]
            }

        stock_features = stock_data[['Volume', 'Open', 'Close', 'High', 'Low', 'SMA_10', 'EMA_10', 'RSI_14']].values
        global_data = stock_features if global_data is None else np.vstack((global_data, stock_features))

    global_scaler.fit(global_data)

    for symbol in stock_symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            continue

        stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
        stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
        stock_data['RSI_14'] = compute_rsi(stock_data)
        stock_data.dropna(inplace=True)

        stock_features = stock_data[['Volume', 'Open', 'Close', 'High', 'Low', 'SMA_10', 'EMA_10', 'RSI_14']].values
        stock_features_scaled = global_scaler.transform(stock_features)

        X, y, stock_ids = create_sequences(stock_features_scaled, stock_mapping[symbol], seq_length)
        all_sequences.append(X)
        all_labels.append(y)
        all_stock_ids.append(stock_ids)

    return np.vstack(all_sequences), np.concatenate(all_labels), np.concatenate(all_stock_ids)

def build_model():
    """Builds the stock prediction model"""
    input_data = Input(shape=(seq_length, 8))
    stock_input = Input(shape=(1,))
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
    return model

# ------------------------- TRAINING -------------------------
if os.path.exists("Stock_Recommendation_Model.h5"):
    print("✅ Loading existing model...")
    model = load_model("Stock_Recommendation_Model.h5", compile=False)
else:
    print("⏳ Loading stock data...")
    X, y, stock_ids = load_stock_data()

    print("🚀 Splitting data...")
    X_train, X_test, y_train, y_test, stock_ids_train, stock_ids_test = train_test_split(
        X, y, stock_ids, test_size=0.2, random_state=42
    )

    print("📈 Building model...")
    model = build_model()

    print("🎯 Training model...")
    history = model.fit(
        [X_train, stock_ids_train], y_train,
        epochs=50, batch_size=32, verbose=1,
        validation_data=([X_test, stock_ids_test], y_test)
    )

    model.save("Stock_Recommendation_Model.h5")
    print("✅ Model saved!")


X, y, stock_ids = load_stock_data()

print("🚀 Splitting data...")
X_train, X_test, y_train, y_test, stock_ids_train, stock_ids_test = train_test_split(
    X, y, stock_ids, test_size=0.2, random_state=42
)

# ------------------------- FLASK ROUTES -------------------------
@app.route('/rankings', methods=['GET'])
def stock_rankings():
    print("🔍 Generating predictions...")
    
    try:
        # Ensure test data exists
        if X_test is None or stock_ids_test is None or len(X_test) == 0:
            print("❌ Error: X_test or stock_ids_test is missing")
            return jsonify({"error": "Data loading issue: X_test or stock_ids_test is None or empty"}), 500

        print(f"✅ X_test shape: {X_test.shape}")
        print(f"✅ stock_ids_test shape: {stock_ids_test.shape}")

        predictions = model.predict([X_test, stock_ids_test])
        predictions = global_scaler.inverse_transform(np.column_stack([np.zeros((predictions.shape[0], 7)), predictions]))[:, -1]
        
        y_test_scaled = global_scaler.inverse_transform(np.column_stack([np.zeros((y_test.shape[0], 7)), y_test]))[:, -1]

        investment_scores = {}
        for symbol in stock_symbols:
            if symbol not in stock_stats:
                print(f"⚠ Warning: Missing stock stats for {symbol}")
                continue
            
            avg_pred = np.mean(predictions[np.where(stock_ids_test == stock_mapping[symbol])])
            mean_return = stock_stats[symbol]['mean_return']
            risk = stock_stats[symbol]['std_dev'] + 1e-6  # Avoid division by zero
            rsi = stock_stats[symbol]['rsi']
            rsi_penalty = 1 - abs((rsi - 50) / 100)  # Smoother RSI adjustment
            
            investment_score = (mean_return / risk) * avg_pred * rsi_penalty  
            investment_scores[symbol] = investment_score

        ranked_stocks = sorted(investment_scores.items(), key=lambda x: x[1], reverse=True)

        print("✅ Stock rankings generated:", ranked_stocks)

        return jsonify({
            "rankings": [
                {"rank": i + 1, "stock": stock, "Investment Score": round(float(score), 2)}
                for i, (stock, score) in enumerate(ranked_stocks)
            ]
        })
    except Exception as e:
        print(f"❌ Error generating rankings: {str(e)}")
        return jsonify({"error": f"Failed to generate rankings: {str(e)}"}), 500


@app.route('/')
def home():
    return render_template('index.html')

# ------------------------- START SERVER -------------------------
if __name__ == '__main__':
    app.run(debug=True)
