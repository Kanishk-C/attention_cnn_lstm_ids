import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2)

        # compute flattened size dynamically
        self._feature_dim = (num_features // 2) * 128

        self.fc1 = nn.Linear(self._feature_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x.squeeze(1)  # logits

