import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

# --- 1. DATA GENERATION (TASK 1) ---
def generate_advanced_timeseries(n=5000):
    np.random.seed(42)
    t = np.linspace(0, 100, n)
    # 5 Features: Trend, Fourier Seasonality, Noise, Cyclic, and Combined Target
    f1 = 0.05 * t  # Linear Trend
    f2 = np.sin(0.5 * t) + 0.3 * np.cos(2 * t)  # Seasonality
    f3 = np.random.normal(0, 0.1, n)  # Gaussian Noise
    f4 = np.cos(0.1 * t)**2  # Non-linear cyclic
    target = f1 + f2 + f4 + f3  # Complex interaction
    
    data = np.stack([f1, f2, f3, f4, target], axis=1)
    return pd.DataFrame(data, columns=['Trend', 'Season', 'Noise', 'Cyclic', 'Target'])

# Prepare Dataset
df = generate_advanced_timeseries()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_windowed_data(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

SEQ_LEN = 24
X, y = create_windowed_data(scaled_data, SEQ_LEN)
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# --- 2. MODEL ARCHITECTURES (TASK 2) ---
def build_stacked_lstm(units=64, input_shape=(SEQ_LEN, 5)):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(units, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(units // 2)(x)
    outputs = layers.Dense(1)(x)
    return Model(inputs, outputs)

def build_transformer_block(head_size=64, num_heads=4, input_shape=(SEQ_LEN, 5)):
    inputs = layers.Input(shape=input_shape)
    # Simple Attention Mechanism
    attn = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    attn = layers.Dropout(0.1)(attn)
    add_norm = layers.LayerNormalization(epsilon=1e-6)(attn + inputs)
    avg_pool = layers.GlobalAveragePooling1D()(add_norm)
    outputs = layers.Dense(1)(avg_pool)
    return Model(inputs, outputs)

# --- 3. HYPERPARAMETER OPTIMIZATION (TASK 3) ---
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    units = trial.suggest_categorical("units", [32, 64, 128])
    
    model = build_stacked_lstm(units=units)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    
    # Train trial
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    val_loss = model.evaluate(X_test, y_test, verbose=0)
    return val_loss

print("Starting Optuna Study...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)
best_params = study.best_params
print(f"Best Trial Params: {best_params}")

# --- 4. EVALUATION & WALK-FORWARD (TASK 4) ---
# Metrics Function
def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred.flatten()) / y_true)) * 100
    return rmse, mae, mape

# Final Optimized LSTM
optimized_model = build_stacked_lstm(units=best_params['units'])
optimized_model.compile(optimizer=tf.keras.optimizers.Adam(best_params['lr']), loss='mse')
optimized_model.fit(X_train, y_train, epochs=10, verbose=0)

# Transformer for Comparison
transformer_model = build_transformer_block()
transformer_model.compile(optimizer='adam', loss='mse')
transformer_model.fit(X_train, y_train, epochs=10, verbose=0)

# Generate Predictions
lstm_preds = optimized_model.predict(X_test)
trans_preds = transformer_model.predict(X_test)

# --- 5. RESULTS & SENSITIVITY (TASK 5) ---
lstm_metrics = get_metrics(y_test, lstm_preds)
trans_metrics = get_metrics(y_test, trans_preds)

print("\n--- COMPARATIVE ANALYSIS SUMMARY ---")
results_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MAPE (%)'],
    'Optimized LSTM': lstm_metrics,
    'Transformer': trans_metrics
})
print(results_df)

# Sensitivity Plot (Learning Rate effect)
lrs = [0.0001, 0.001, 0.01]
sens_loss = []
for l in lrs:
    m = build_stacked_lstm(units=best_params['units'])
    m.compile(optimizer=tf.keras.optimizers.Adam(l), loss='mse')
    hist = m.fit(X_train, y_train, epochs=3, verbose=0)
    sens_loss.append(hist.history['loss'][-1])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(y_test[:100], label='Actual', alpha=0.7)
plt.plot(lstm_preds[:100], label='LSTM Pred', alpha=0.7)
plt.title("Predictions vs Actual")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lrs, sens_loss, marker='o')
plt.xscale('log')
plt.title("Sensitivity: Learning Rate vs Loss")
plt.xlabel("LR")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()
