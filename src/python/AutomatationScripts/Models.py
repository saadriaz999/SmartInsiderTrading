import torch
from torch import nn


# Define the LSTM models
class StockLSTM(nn.Module):
    def __init__(self):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=4, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last timestep output
        return out


class CombinedLSTM(nn.Module):
    def __init__(self):
        super(CombinedLSTM, self).__init__()
        self.lstm_stock = nn.LSTM(input_size=5, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        self.lstm_insider = nn.LSTM(input_size=3, hidden_size=50, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(100, 1)

    def forward(self, stock_input, insider_input):
        _, (stock_hidden, _) = self.lstm_stock(stock_input)
        _, (insider_hidden, _) = self.lstm_insider(insider_input)
        combined = torch.cat((stock_hidden[-1], insider_hidden[-1]), dim=1)
        output = self.fc(combined)
        return output
