"""
Script to train and save the water level prediction model.
Run this script to generate the model.pkl file.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle
import os

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load and preprocess data
print("Loading data...")
df = pd.read_excel(os.path.join(SCRIPT_DIR, "data/godavari_adhala_adhala_mahasw_man_anantwadi.xlsx"))

# Process datetime and rename columns
df['Date Time'] = pd.to_datetime(df['Data Time'])
df = df.sort_values('Date Time')
df.rename(columns={'Date Time': 'datetime', 'Data Value': 'level'}, inplace=True)

# Set frequency and interpolate
df = df.set_index('datetime').asfreq('9h')
df['level'] = df['level'].interpolate()

# Create sequences
def create_sequences(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        seq = series[i:i+window_size]
        label = series[i+window_size]
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

window_size = 10
X, y = create_sequences(df['level'].values, window_size)

# Train-test split
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Scale data
print("Scaling data...")
scaler = MinMaxScaler()
scaler.fit(y_train.reshape(-1, 1))

X_train_scaled = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
y_train_scaled = scaler.transform(y_train.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

# Build model
print("Building LSTM model...")
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        64,
        activation='tanh',
        return_sequences=False,
        input_shape=(window_size, 1)
    ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train model
print("Training model...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"\n--- Model Performance ---")
print(f"RMSE: {rmse:.3f} m")
print(f"MAE: {mae:.3f} m")
print(f"R² Score: {r2:.3f}")

# Save model and scaler
print("\nSaving model to model.pkl...")
model.save(os.path.join(SCRIPT_DIR, "lstm_model.keras"))

model_data = {
    'scaler': scaler,
    'window_size': window_size,
    'model_path': 'lstm_model.keras',
    'rmse': rmse,
    'mae': mae,
    'r2': r2
}

with open(os.path.join(SCRIPT_DIR, "model.pkl"), 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved successfully!")
print(f"Files created:")
print(f"  - model.pkl (scaler and metadata)")
print(f"  - lstm_model.keras (trained LSTM model)")
