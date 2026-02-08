import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(self, num_features, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(-1)  # (batch, features, 1)

        out, _ = self.lstm(x)  # (batch, features, hidden)
        out = out[:, -1, :]    # last time step

        out = self.dropout(out)
        out = self.fc(out)

        return out.squeeze(1)  # logits

