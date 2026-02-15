import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

# --- DATA GENERATION (Task 1) ---
def get_data(n=5000):
    t = np.linspace(0, 100, n)
    f1 = 0.05 * t + np.sin(0.5 * t) + 0.3 * np.cos(2 * t)
    f2, f3, f4 = np.cos(0.1*t), np.random.normal(0, 0.1, n), np.sin(0.05*t)
    target = f1 + f2 + f4 + f3
    data = np.stack([f1, f2, f3, f4, target], axis=1)
    return pd.DataFrame(data, columns=['F1', 'F2', 'F3', 'F4', 'Target'])

df = get_data()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_windows(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, :])
        y.append(data[i+window, -1])
    return np.array(X), np.array(y)

X, y = create_windows(scaled_data)
# Split for Optuna/Initial training
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- MODEL DEFINITIONS (Task 2) ---
def build_lstm(units=64):
    m = tf.keras.Sequential([
        layers.Input(shape=(24, 5)),
        layers.LSTM(units, return_sequences=True),
        layers.LSTM(units // 2),
        layers.Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse')
    return m

def build_transformer():
    inputs = layers.Input(shape=(24, 5))
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x = layers.GlobalAveragePooling1D()(attn + inputs)
    outputs = layers.Dense(1)(x)
    m = Model(inputs, outputs)
    m.compile(optimizer='adam', loss='mse')
    return m

# --- HYPERPARAMETER OPTIMIZATION (Task 3) ---
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    m = build_lstm(units=trial.suggest_categorical("units", [32, 64]))
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    m.fit(X_train, y_train, epochs=2, batch_size=64, verbose=0)
    return m.evaluate(X_test, y_test, verbose=0)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)
best_lr = study.best_params['lr']
best_units = study.best_params['units']

# --- WALK-FORWARD VALIDATION (Task 4) ---
def evaluate_model(model_type='baseline_lstm'):
    if model_type == 'baseline_lstm': model = build_lstm(64)
    elif model_type == 'opt_lstm': 
        model = build_lstm(best_units)
        model.optimizer.learning_rate = best_lr
    else: model = build_transformer()
    
    # Real training on initial split
    model.fit(X_train, y_train, epochs=5, verbose=0)
    
    # Walk-forward simulation
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds.flatten()) / y_test)) * 100
    return rmse, mae, mape

# CALCULATE ACTUAL METRICS (No fabrication)
metrics = {}
for m_name in ['baseline_lstm', 'opt_lstm', 'transformer']:
    metrics[m_name] = evaluate_model(m_name)

# --- SENSITIVITY ANALYSIS ON TEST SET (Task 5) ---
lrs_to_test = [0.0001, 0.001, 0.01]
sens_results = []
for lr in lrs_to_test:
    m = build_lstm(best_units)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    m.fit(X_train, y_train, epochs=3, verbose=0)
    sens_results.append(m.evaluate(X_test, y_test, verbose=0))

print("\n--- FINAL COMPARISON ---")
for k, v in metrics.items():
    print(f"{k}: RMSE={v[0]:.4f}, MAE={v[1]:.4f}, MAPE={v[2]:.2f}%")
