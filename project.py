import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras import layers, Model
import optuna
import warnings

warnings.filterwarnings('ignore')

# --- 1. DATA GENERATION (Simulated EV Population Growth) ---
def generate_ev_data(samples=1000):
    np.random.seed(42)
    t = np.arange(samples)
    # Simulate non-linear growth + seasonality + noise
    base_growth = 100 * np.exp(0.005 * t) 
    seasonality = 50 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 20, samples)
    
    target = base_growth + seasonality + noise
    # Add 4 additional features for multivariate analysis
    f1 = base_growth * 0.1 + np.random.normal(0, 5, samples)
    f2 = np.random.uniform(10, 20, samples)
    f3 = np.sin(2 * np.pi * t / 52) * 10
    f4 = np.random.normal(50, 2, samples)
    
    data = np.stack([f1, f2, f3, f4, target], axis=1)
    return pd.DataFrame(data, columns=['Charging_Stations', 'Subsidies', 'Cyclic_Factor', 'Energy_Cost', 'EV_Population'])

df = generate_ev_data()

# --- 2. PREPROCESSING ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, -1])
    return np.array(X), np.array(y)

# Default sequence length
LOOKBACK = 12
X, y = create_sequences(scaled_data, LOOKBACK)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- 3. BASELINE MODEL (Exponential Smoothing) ---
# Holt-Winters on the target variable only
train_target = df['EV_Population'][:split]
test_target = df['EV_Population'][split:]

baseline_model = ExponentialSmoothing(train_target, seasonal='add', seasonal_periods=12).fit()
baseline_preds = baseline_model.forecast(len(test_target))

def get_metrics(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape

baseline_metrics = get_metrics(test_target, baseline_preds)

# --- 4. DEEP LEARNING MODEL & HYPERPARAMETER OPTIMIZATION (Optuna) ---
def build_model(trial):
    # 5 Hyperparameters to search
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units = trial.suggest_int('units', 32, 128)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    # Note: seq_length is often tuned but requires re-windowing data. 
    # To keep code stable, we tune 'activation' as the 5th param.
    act = trial.suggest_categorical('activation', ['relu', 'tanh'])

    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = inputs
    for i in range(n_layers):
        return_seq = True if i < n_layers - 1 else False
        x = layers.LSTM(units, return_sequences=return_seq, activation=act)(x)
        x = layers.Dropout(dropout)(x)
    
    outputs = layers.Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

def objective(trial):
    model = build_model(trial)
    # Fast training for search
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    loss = model.evaluate(X_test, y_test, verbose=0)
    return loss

print("Starting Bayesian Optimization (Optuna)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10) # Set to 20+ for even better results

print("\n--- OPTIMIZATION LOG ---")
print(f"Best Trial Score (MSE): {study.best_value:.6f}")
print(f"Best Hyperparameters: {study.best_params}")

# --- 5. FINAL MODEL TRAINING ---
best_model = build_model(study.best_trial)
# More epochs for the final model
best_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Rescale predictions back to original scale
y_pred_scaled = best_model.predict(X_test)
# Simple inverse scaling for the target column only
y_test_unscaled = df['EV_Population'].values[split + LOOKBACK:]
y_pred_unscaled = (y_pred_scaled.flatten() * (df['EV_Population'].max() - df['EV_Population'].min())) + df['EV_Population'].min()

optimized_metrics = get_metrics(y_test_unscaled, y_pred_unscaled)

# --- 6. FINAL COMPARISON TABLE ---
results = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MAPE (%)'],
    'Exponential Smoothing (Baseline)': baseline_metrics,
    'Optimized LSTM': optimized_metrics
})

print("\n--- FINAL COMPARISON TABLE ---")
print(results.to_string(index=False))
