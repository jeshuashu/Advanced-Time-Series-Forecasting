import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.forecasting.theta import ThetaModel

# ==========================================
# TASK 1: SYNTHETIC DATA GENERATION (5 YEARS)
# ==========================================
def generate_complex_data():
    np.random.seed(42)
    # 5 years of hourly data = 5 * 365 * 24 = 43,800 points
    periods = 5 * 365 * 24
    time = np.arange(periods)
    
    # 1. Non-stationary trend (linear + slight exponential)
    trend = 0.005 * time + 0.000001 * (time**2)
    
    # 2. Daily seasonality (24h)
    daily = 10 * np.sin(2 * np.pi * time / 24)
    
    # 3. Weekly seasonality (168h)
    weekly = 15 * np.cos(2 * np.pi * time / 168)
    
    # 4. Noise
    noise = np.random.normal(0, 2, periods)
    
    series = 50 + trend + daily + weekly + noise
    return series

data = generate_complex_data().reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length, pred_length):
    x, y = [], []
    for i in range(len(data) - seq_length - pred_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_length])
    return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y))

# Params
SEQ_LENGTH = 168  # 1 week look-back
PRED_LENGTH = 24  # 1 day forecast
X, y = create_sequences(scaled_data, SEQ_LENGTH, PRED_LENGTH)

# ==========================================
# TASK 2: ATTENTION-BASED LSTM MODEL
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2SeqAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        out = self.fc(context)
        return out, weights

model = Seq2SeqAttention(1, 64, PRED_LENGTH)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# TASK 3 & 4: TRAINING & EVALUATION
# ==========================================
# (Simplified training loop for demo; in production use TimeSeriesSplit)
model.train()
for epoch in range(5): # Increase epochs for better score
    optimizer.zero_grad()
    output, _ = model(X[:1000]) # Training on subset for speed
    loss = criterion(output, y[:1000].squeeze(-1))
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0: print(f"Epoch {epoch} Loss: {loss.item():.6f}")

# Visualizing Attention
model.eval()
with torch.no_grad():
    sample_x = X[-1].unsqueeze(0)
    pred, weights = model(sample_x)
    
plt.figure(figsize=(10, 4))
plt.plot(weights.squeeze().numpy())
plt.title("Attention Weights over 168-hour Input Sequence")
plt.xlabel("Hours in Past")
plt.ylabel("Importance Weight")
plt.show()
