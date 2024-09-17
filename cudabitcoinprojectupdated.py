import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from optuna.samplers import TPESampler
from datetime import datetime, timedelta

# Fetch historical Bitcoin data using yfinance
def fetch_bitcoin_data(start_date=None, end_date=None):
    if end_date is None:
        end_date = datetime.today()
    else:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    if start_date is None:
        start_date = (end_date - timedelta(days=1825)).strftime('%Y-%m-%d')
    else:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
    return btc_data

# Feature engineering: Add additional technical indicators
def add_technical_indicators(data):
    data = data.copy()

    # Ensure 'price' column exists
    if 'price' not in data.columns:
        raise ValueError("The 'price' column is not present in the DataFrame.")

    # Calculate RSI
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD and Signal
    ema12 = data['price'].ewm(span=12, adjust=False).mean()
    ema26 = data['price'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

    # Bollinger Bands
    rolling_mean = data['price'].rolling(window=20).mean()
    rolling_std = data['price'].rolling(window=20).std()
    data['Bollinger_High'] = rolling_mean + (rolling_std * 2)
    data['Bollinger_Low'] = rolling_mean - (rolling_std * 2)

    # ATR (Average True Range)
    high_low = data['price'].rolling(window=14).max() - data['price'].rolling(window=14).min()
    high_close = np.abs(data['price'].rolling(window=14).max() - data['price'].shift(1))
    low_close = np.abs(data['price'].rolling(window=14).min() - data['price'].shift(1))
    tr = high_low.combine(high_close, max).combine(low_close, max)
    data['ATR'] = tr.rolling(window=14).mean()

    # On-Balance Volume (OBV)
    data['OBV'] = (np.sign(data['price'].diff()) * data['Volume']).cumsum()

    return data

# Fetch Bitcoin data and add features
bitcoin_data = fetch_bitcoin_data()
bitcoin_data = bitcoin_data[['Close', 'Volume']]
bitcoin_data.columns = ['price', 'Volume']
bitcoin_data = add_technical_indicators(bitcoin_data)

# Handle NaN values and normalization
print("Before handling NaNs:", bitcoin_data.isna().sum())
bitcoin_data.fillna(method='ffill', inplace=True)
bitcoin_data.fillna(0, inplace=True)
print("After filling NaNs:", bitcoin_data.isna().sum())

# Normalize the data with StandardScaler (for better handling of outliers)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(bitcoin_data)

# Prepare the data for the LSTM model
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length, 0]  # Predicting the 'price' column
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

SEQ_LENGTH = 240  # Increased sequence length for longer trend capture
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split the data
validation_size = int(len(X) * 0.30)
train_size = len(X) - validation_size

X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Convert the data into PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().unsqueeze(1).to(device)
X_val = torch.from_numpy(X_val).float().to(device)
y_val = torch.from_numpy(y_val).float().unsqueeze(1).to(device)

# Define DataLoader with mini-batches
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the LSTM Model with Dropout and Regularization
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_layer_size=150, dropout=0.4, output_size=1, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Hyperparameter tuning with Optuna
def objective(trial):
    hidden_layer_size = trial.suggest_int('hidden_layer_size', 100, 400)
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    model = LSTMModel(input_size=X_train.shape[2], hidden_layer_size=hidden_layer_size, dropout=dropout).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    EPOCHS = 100
    patience = 15
    best_val_loss = np.inf
    epochs_without_improvement = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred_train = model(X_batch)
            loss = loss_function(y_pred_train, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch_val, y_batch_val in val_loader:
                y_pred_val = model(X_batch_val)
                val_loss += loss_function(y_pred_val, y_batch_val).item()

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {running_loss / len(train_loader):.4f}, Val Loss = {val_loss / len(val_loader):.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping in trial triggered at epoch {epoch}.")
            break

    return best_val_loss

# Perform hyperparameter tuning with TPESampler
print("Starting hyperparameter tuning...")
sampler = TPESampler()
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=30)
print(f'Best hyperparameters: {study.best_params}')

# Train the final model with the best hyperparameters
best_params = study.best_params
model = LSTMModel(input_size=X_train.shape[2], hidden_layer_size=best_params['hidden_layer_size'], dropout=best_params['dropout']).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'])
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

EPOCHS = 100
patience = 15
best_val_loss = np.inf
epochs_without_improvement = 0

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred_train = model(X_batch)
        loss = loss_function(y_pred_train, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch_val, y_batch_val in val_loader:
            y_pred_val = model(X_batch_val)
            val_loss += loss_function(y_pred_val, y_batch_val).item()

    scheduler.step(val_loss)

    if epoch % 10 == 0:
        print(f'Epoch {epoch} - Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Saving model at epoch {epoch} with Val Loss: {best_val_loss:.4f}')
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss:.4f}")
        break

# Load the best model and make predictions
print("Loading the best model and making predictions...")
model.load_state_dict(torch.load('best_model.pth', weights_only=True))

model.eval()
with torch.no_grad():
    y_pred_val = model(X_val)

# Inverse transform the predictions
y_train_unscaled = scaler.inverse_transform(np.concatenate((y_train.cpu().detach().numpy().reshape(-1,1), np.zeros((y_train.shape[0], 7))), axis=1))[:,0]
y_val_unscaled = scaler.inverse_transform(np.concatenate((y_val.cpu().detach().numpy().reshape(-1,1), np.zeros((y_val.shape[0], 7))), axis=1))[:,0]
y_pred_val_unscaled = scaler.inverse_transform(np.concatenate((y_pred_val.cpu().detach().numpy().reshape(-1,1), np.zeros((y_pred_val.shape[0], 7))), axis=1))[:,0]

# Predict the Next 30 Days with dynamically updated indicators
print("Predicting the next 30 days...")
future_predictions = []
last_sequence = X_val[-1].unsqueeze(0).to(device)

# Use recent historical data to initialize indicators
recent_prices = bitcoin_data['price'].iloc[-14:].values  # Using last 14 days to initialize RSI
macd = bitcoin_data['MACD'].iloc[-1]
macd_signal = bitcoin_data['MACD_signal'].iloc[-1]
bollinger_mean = bitcoin_data['price'].rolling(window=20).mean().iloc[-1]
bollinger_std = bitcoin_data['price'].rolling(window=20).std().iloc[-1]
atr = bitcoin_data['ATR'].iloc[-1]  # Initialize ATR from the last value

for day in range(30):
    with torch.no_grad():
        next_price = model(last_sequence)
    
    predicted_price = next_price.item()
    future_predictions.append(predicted_price)

    # Update recent prices and indicators
    recent_prices = np.append(recent_prices[1:], predicted_price)  # Update recent prices for indicators
    
    # Update RSI
    delta = np.diff(recent_prices)
    gain = np.mean(delta[delta > 0]) if len(delta[delta > 0]) > 0 else 0
    loss = np.mean(-delta[delta < 0]) if len(delta[delta < 0]) > 0 else 0
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    last_rsi = 100 - (100 / (1 + rs))

    # Update MACD
    macd = (macd * 25 + predicted_price) / 26  # Exponential smoothing for MACD
    macd_signal = (macd_signal * 8 + macd) / 9  # Update MACD signal
    macd_histogram = macd - macd_signal  # MACD histogram

    # Update Bollinger Bands
    bollinger_mean = (bollinger_mean * 19 + predicted_price) / 20
    bollinger_std = (bollinger_std * 19 + abs(predicted_price - bollinger_mean)) / 20
    bollinger_upper = bollinger_mean + (2 * bollinger_std)
    bollinger_lower = bollinger_mean - (2 * bollinger_std)

    # Update ATR
    true_range = max(predicted_price - recent_prices[-2], abs(predicted_price - recent_prices[-2]))
    atr = (atr * 13 + true_range) / 14

    # Construct the new input with updated features
    new_input = [
        predicted_price,  # Price
        last_rsi,  # RSI
        macd,  # MACD
        macd_signal,  # MACD Signal
        macd_histogram,  # MACD Histogram
        bollinger_upper,  # Bollinger Upper Band
        bollinger_lower,  # Bollinger Lower Band
        atr  # ATR
    ]
    
    new_input_tensor = torch.tensor(new_input).unsqueeze(0).unsqueeze(0).float().to(device)

    # Update the input sequence
    last_sequence = torch.cat((last_sequence[:, 1:, :], new_input_tensor), dim=1)

    # Now inverse transform the predicted price to get the actual Bitcoin price
    predicted_price_scaled = np.array([[predicted_price] + [0]*7])  # Add 7 zeros to match feature count
    predicted_price_actual = scaler.inverse_transform(predicted_price_scaled)[0, 0]  # Inverse transform
    
    print(f"Day {day+1}: Scaled Predicted Price = {predicted_price:.2f}, Actual Predicted Price = {predicted_price_actual:.2f}")

# Convert predictions to a DataFrame for visualization
future_dates = pd.date_range(start=bitcoin_data.index[-1] + pd.Timedelta(days=1), periods=30)
future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Scaled_Predicted_Price'])

# Convert scaled future predictions to actual prices
future_actual_prices = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1,1), 
                                                                np.zeros((len(future_predictions), 7))), axis=1))[:,0]
future_df['Actual_Predicted_Price'] = future_actual_prices

# Display actual future predictions for the next 30 days
print(future_df[['Actual_Predicted_Price']])

# Align validation predictions to the correct index length
validation_start_index = len(bitcoin_data) - len(y_val_unscaled)
validation_indices = bitcoin_data.index[validation_start_index:validation_start_index + len(y_pred_val_unscaled)]

# Set up the figure and axis
plt.figure(figsize=(14, 7))

# Plot the historical prices for the last 5 years (1855 days)
plt.plot(bitcoin_data.index[-1855:], bitcoin_data['price'][-1855:], label='Historical Price (Last 1855 Days 5yrs+)')

# Plot the validation predictions with the correct dates
plt.plot(validation_indices, y_pred_val_unscaled, label='Validation Predictions', linestyle='--')

# Plot the future predictions for the next 30 days
plt.plot(future_df.index, future_df['Actual_Predicted_Price'], label='Future Predictions (Next 30 Days)', linestyle='--', color='red')

# Format the x-axis to show dates with better precision
ax = plt.gca()

# Use AutoDateLocator and DateFormatter for better date representation
locator = mdates.AutoDateLocator()
formatter = mdates.DateFormatter('%b %d, %Y')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Rotate the date labels for better readability
plt.gcf().autofmt_xdate()

# Set the labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Prediction (Last 5 years and Future 30 Days)')

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Evaluate the model
mse = mean_squared_error(y_val_unscaled, y_pred_val_unscaled)
mae = mean_absolute_error(y_val_unscaled, y_pred_val_unscaled)
mape = mean_absolute_percentage_error(y_val_unscaled, y_pred_val_unscaled)
r2 = r2_score(y_val_unscaled, y_pred_val_unscaled)

print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')
print(f'R² Score: {r2:.4f}')
