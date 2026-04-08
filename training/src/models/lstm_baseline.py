import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Baseline LSTM that treats the 76 features as a sequence
    of 76 time steps, each with 1 feature.
    """
    def __init__(self, num_features, hidden_size=64):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=1, hidden_size=hidden_size,
                               num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)           # (B, F, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]           # last time step
        return self.fc(self.dropout(out)).squeeze(1)
