import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCNNLSTM(nn.Module):
    """
    Attention-CNN-LSTM: adds a self-attention layer on top of
    the CNN-LSTM hybrid so the model can focus on the most
    informative features dynamically per sample.

    Architecture:
        Input (76 features)
        → Conv1D (64, k=3) + BN + ReLU + MaxPool
        → Conv1D (128, k=3) + BN + ReLU
        → LSTM (64 hidden)
        → Scaled-dot-product self-attention (over LSTM outputs)
        → Dropout (0.3)
        → Dense → binary logit
    """
    def __init__(self, num_features):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(kernel_size=2)

        # LSTM
        self.lstm    = nn.LSTM(input_size=128, hidden_size=64,
                               num_layers=1, batch_first=True)

        # Attention — scalar score per time step
        self.attn_fc = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)                            # (B, 1, F)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))            # (B, 128, seq)
        x = x.permute(0, 2, 1)                        # (B, seq, 128)

        lstm_out, _ = self.lstm(x)                    # (B, seq, 64)

        # Attention weights over the sequence dimension
        scores  = self.attn_fc(lstm_out)              # (B, seq, 1)
        weights = torch.softmax(scores, dim=1)        # normalise
        context = torch.sum(weights * lstm_out, dim=1)  # (B, 64)

        return self.fc(self.dropout(context)).squeeze(1)
