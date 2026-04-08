"""
src/utils/train_utils.py  —  CPU-optimised training helpers.

Works on Windows, Linux, macOS, Google Colab.
For GPU training use the functions defined in IDS_Colab_Notebook.ipynb.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, classification_report
)

# Use all available CPU cores for PyTorch linear algebra
_cores = os.cpu_count() or 4
torch.set_num_threads(_cores)
torch.set_num_interop_threads(max(1, _cores // 2))


def load_data():
    """Load preprocessed .npy files. Uses os.path.join for cross-platform paths."""
    base = os.path.join("data", "processed")
    return (
        torch.tensor(np.load(os.path.join(base, "X_train.npy")), dtype=torch.float32),
        torch.tensor(np.load(os.path.join(base, "y_train.npy")), dtype=torch.float32),
        torch.tensor(np.load(os.path.join(base, "X_test.npy")),  dtype=torch.float32),
        torch.tensor(np.load(os.path.join(base, "y_test.npy")),  dtype=torch.float32),
    )


def train_model(model, X_train, y_train, epochs=10, batch_size=256, lr=1e-3):
    """
    CPU training loop with OneCycleLR for faster convergence.

    num_workers=0  → safe on Windows (no multiprocessing fork issues)
    pin_memory=False → no benefit without a GPU
    batch_size=256   → CPU BLAS sweet spot (larger than GPU default of 32-64)
    set_to_none=True → skips zeroing gradient tensors, slightly faster
    """
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        steps_per_epoch=len(loader),
        epochs=epochs,
        pct_start=0.3,
    )

    history = {"loss": []}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        avg = total / len(loader)
        history["loss"].append(avg)
        print(f"  Epoch {epoch+1:02d}/{epochs}  loss={avg:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    return history


def evaluate_model(model, X_test, y_test, batch_size=512):
    """
    Batched evaluation using torch.inference_mode (faster than no_grad).
    Batching prevents OOM on large test sets.
    zero_division=0 suppresses sklearn warnings on edge-case predictions.
    """
    model.eval()
    all_preds = []

    loader = DataLoader(
        TensorDataset(X_test),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )

    with torch.inference_mode():
        for (xb,) in loader:
            all_preds.append((torch.sigmoid(model(xb)) > 0.5).int())

    preds  = torch.cat(all_preds).numpy()
    y_true = y_test.numpy().astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "f1":       f1_score(y_true, preds, zero_division=0),
        "mcc":      matthews_corrcoef(y_true, preds),
    }

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")
    print(f"  MCC      : {metrics['mcc']:.4f}")
    print()
    print(classification_report(y_true, preds,
                                target_names=["BENIGN", "ATTACK"],
                                zero_division=0))
    return metrics, preds, y_true
