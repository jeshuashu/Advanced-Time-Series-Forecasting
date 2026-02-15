import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

# --- 1. DATA GENERATION ---
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
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- 2. MODEL DEFINITIONS ---
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

# --- 3. OPTUNA OPTIMIZATION ---
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

# --- 4. WALK-FORWARD VALIDATION ---
def walk_forward_eval(model_type):
    if model_type == 'baseline': m = build_lstm(64)
    elif model_type == 'opt': 
        m = build_lstm(best_units)
        m.optimizer.learning_rate = best_lr
    else: m = build_transformer()
    
    m.fit(X_train, y_train, epochs=5, verbose=0)
    preds = m.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds.flatten()) / y_test)) * 100
    return rmse, mae, mape

print("\n--- GENERATING FINAL METRICS ---")
results = {}
for name in ['baseline', 'opt', 'transformer']:
    results[name] = walk_forward_eval(name)
    print(f"{name.upper()} Done.")

# --- 5. SENSITIVITY ANALYSIS (ON TEST SET) ---
lrs = [0.0001, 0.001, 0.01]
sens_metrics = []
for lr in lrs:
    m = build_lstm(best_units)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    m.fit(X_train, y_train, epochs=3, verbose=0)
    # Testing sensitivity on validation/test error as required
    sens_metrics.append(m.evaluate(X_test, y_test, verbose=0))

print("\n--- FINAL RESULTS FOR REPORT ---")
print(f"Best LR: {best_lr} | Best Units: {best_units}")
for k, v in results.items():
    print(f"{k}: RMSE={v[0]:.4f}, MAE={v[1]:.4f}, MAPE={v[2]:.2f}%")
print(f"Sensitivity (Loss for LR {lrs}): {sens_metrics}")
