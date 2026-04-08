import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    """
    1D-CNN baseline for tabular intrusion detection.
    Treats the 76 features as a 1D signal and applies
    two convolutional layers followed by a dense classifier.
    """
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(kernel_size=2)

        flat_size    = (num_features // 2) * 128
        self.fc1     = nn.Linear(flat_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)                           # (B, 1, F)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(1)
