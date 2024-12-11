import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

from Models import StockLSTM, CombinedLSTM

# Function to preprocess data
def preprocess_data(insider_path, stock_path):
    insider_data = pd.read_csv(insider_path).drop(columns=['company_name', 'ticker', 'insider_name', 'job_title', 'trade_type', 'final_shares_owned', 'change_in_shares_owned'])
    stock_data = pd.read_csv(stock_path)

    insider_data['filing_date'] = pd.to_datetime(insider_data['filing_date'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.tz_localize(None)

    # Merge and align data
    insider_data = insider_data.groupby("filing_date", as_index=False).agg({
        "price": "mean", "quantity_traded": "sum", "value_of_shares_traded": "sum"}).set_index("filing_date").reindex(
        stock_data["Date"].dt.date).fillna(0).reset_index()

    insider_data['Date'] = pd.to_datetime(insider_data['Date'])

    merged_data = pd.merge(
        stock_data, insider_data, left_on="Date", right_on="Date", how="left")

    return stock_data, insider_data, merged_data

# Function to create sequences
def create_sequences(data, target_col, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 1])  # Use raw data directly
        y.append(data[i, target_col])
    return np.array(X), np.array(y)

# Train and evaluate models
def train_and_evaluate(stock_data, insider_data, merged_data, sequence_length=60, epochs=1):
    # Use raw data directly
    raw_stock = stock_data.drop(columns=['Date']).values
    raw_merged = merged_data.drop(columns=['Date']).values

    # Stock-only model
    X_stock, y_stock = create_sequences(raw_stock, 4, sequence_length)  # 'Close' is column 4
    train_size = int(len(X_stock) * 0.99)
    X_stock = X_stock.reshape((X_stock.shape[0], X_stock.shape[1], 1))
    X_stock_train, X_stock_test = torch.tensor(X_stock[:train_size], dtype=torch.float32), torch.tensor(X_stock[train_size:], dtype=torch.float32)
    y_stock_train, y_stock_test = torch.tensor(y_stock[:train_size], dtype=torch.float32), torch.tensor(y_stock[train_size:], dtype=torch.float32)

    stock_model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(stock_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        print('z')
        stock_model.train()
        optimizer.zero_grad()
        output = stock_model(X_stock_train)
        loss = criterion(output.squeeze(), y_stock_train)
        loss.backward()
        optimizer.step()

    stock_model.eval()
    with torch.no_grad():
        predictions = stock_model(X_stock_test).squeeze().numpy()
        actuals = y_stock_test.numpy()

    mse_stock = mean_squared_error(actuals, predictions)
    mae_stock = mean_absolute_error(actuals, predictions)
    mape_stock = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    # Combined model
    X_combined_stock, X_combined_insider, y_combined = [], [], []
    for i in range(sequence_length, len(raw_merged)):
        X_combined_stock.append(raw_merged[i-sequence_length:i, :5])  # First 5 columns: stock features
        X_combined_insider.append(raw_merged[i-sequence_length:i, 5:])  # Next columns: insider features
        y_combined.append(raw_merged[i, 4])  # Target: 'Close'

    X_combined_stock, X_combined_insider, y_combined = np.array(X_combined_stock), np.array(X_combined_insider), np.array(y_combined)
    train_size = int(len(X_combined_stock) * 0.99)
    X_combined_stock_train, X_combined_stock_test = torch.tensor(X_combined_stock[:train_size], dtype=torch.float32), torch.tensor(X_combined_stock[train_size:], dtype=torch.float32)
    X_combined_insider_train, X_combined_insider_test = torch.tensor(X_combined_insider[:train_size], dtype=torch.float32), torch.tensor(X_combined_insider[train_size:], dtype=torch.float32)
    y_combined_train, y_combined_test = torch.tensor(y_combined[:train_size], dtype=torch.float32), torch.tensor(y_combined[train_size:], dtype=torch.float32)

    combined_model = CombinedLSTM()
    optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        print('y')
        combined_model.train()
        optimizer.zero_grad()
        output = combined_model(X_combined_stock_train, X_combined_insider_train)
        print(output)
        loss = criterion(output.squeeze(), y_combined_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(combined_model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()

    combined_model.eval()
    with torch.no_grad():
        predictions = combined_model(X_combined_stock_test, X_combined_insider_test).squeeze().numpy()
        actuals = y_combined_test.numpy()
        print(predictions)
        print(actuals)

    mse_combined = mean_squared_error(actuals, predictions)
    mae_combined = mean_absolute_error(actuals, predictions)
    mape_combined = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    return {"model_1_mse": mse_stock, "model_1_mae": mae_stock, "model_1_mape": mape_stock,
            "model_2_mse": mse_combined, "model_2_mae": mae_combined, "model_2_mape": mape_combined}
