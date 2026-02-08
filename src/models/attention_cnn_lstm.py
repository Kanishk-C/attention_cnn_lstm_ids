import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCNNLSTM(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # Attention
        self.attn_fc = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))

        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention weights
        attn_scores = self.attn_fc(lstm_out)          # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)

        x = self.dropout(context)
        x = self.fc(x)

        return x.squeeze(1)  # logits

