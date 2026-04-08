import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM: CNN extracts spatial features from the
    feature vector, then LSTM models temporal dependencies
    across the CNN feature maps.
    """
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.lstm  = nn.LSTM(input_size=128, hidden_size=64,
                             num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)                            # (B, 1, F)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)                        # (B, seq, 128)
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1])).squeeze(1)
