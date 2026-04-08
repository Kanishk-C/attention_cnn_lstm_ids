import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from src.models.cnn_lstm import CNNLSTM

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

def load_data():
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

def train():
    X_train, y_train, X_test, y_test = load_data()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = CNNLSTM(num_features=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.6f}")

    evaluate(model, X_test, y_test)

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = (torch.sigmoid(logits) > 0.5).int().numpy()

    y_true = y_test.numpy().astype(int)

    print("\nEvaluation Results:")
    print("Accuracy:", accuracy_score(y_true, preds))
    print("F1-score:", f1_score(y_true, preds))
    print("MCC:", matthews_corrcoef(y_true, preds))

if __name__ == "__main__":
    train()

