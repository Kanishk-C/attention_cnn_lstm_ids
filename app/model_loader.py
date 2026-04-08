"""
model_loader.py

Loads a trained model checkpoint (.pt file) and the fitted scaler,
then exposes a simple classify() function used by the live dashboard.

FIX (2026-04-08):
    num_features is now AUTO-DETECTED from the saved scaler so the
    live demo always matches whatever feature count was used at training
    time (78 from raw CIC-IDS2017 or 76 after manual drops, etc.).
    Pass num_features=None (default) to use the scaler value.
"""

import os
import sys
import warnings
import numpy as np
import torch
import joblib

# Scaler was fitted on a DataFrame with named columns; we pass numpy arrays.
# The transform is still positionally correct — suppress the noise.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_HERE)
_TRAINING = os.path.join(_ROOT, "training")

for _p in (_ROOT, _TRAINING):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.cnn_baseline       import CNNBaseline
from src.models.lstm_baseline      import LSTMBaseline
from src.models.cnn_lstm           import CNNLSTM
from src.models.attention_cnn_lstm import AttentionCNNLSTM

MODEL_CLASSES = {
    "CNN":                CNNBaseline,
    "LSTM":               LSTMBaseline,
    "CNN-LSTM":           CNNLSTM,
    "Attention-CNN-LSTM": AttentionCNNLSTM,
}

CKPT_DIR    = os.path.join(_ROOT, "experiments", "checkpoints")
SCALER_PATH = os.path.join(_ROOT, "data", "processed", "scaler.pkl")
FEAT_PATH   = os.path.join(_ROOT, "data", "processed", "feature_names.txt")


class IDS_Model:
    """
    Wraps a trained IDS model checkpoint for live inference.

    Parameters
    ----------
    model_name   : str   — "CNN", "LSTM", "CNN-LSTM", "Attention-CNN-LSTM"
    num_features : int or None — pass None to auto-detect from scaler (recommended)
    threshold    : float — probability above this → ATTACK  (default 0.5)
    """

    def __init__(self, model_name: str = "CNN-LSTM",
                 num_features=None, threshold: float = 0.5):
        self.model_name = model_name
        self.threshold  = threshold

        # ── Load scaler ───────────────────────────────────────────────
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
        else:
            self.scaler = None
            print(f"[WARN] Scaler not found at {SCALER_PATH}")

        # ── Auto-detect feature count from scaler ─────────────────────
        if num_features is None:
            if self.scaler is not None and hasattr(self.scaler, "n_features_in_"):
                self.num_features = int(self.scaler.n_features_in_)
                print(f"[INFO] Auto-detected {self.num_features} features from scaler")
            else:
                self.num_features = 76
                print("[WARN] Scaler feature count unavailable — defaulting to 76")
        else:
            # Honour explicit value but warn on mismatch
            scaler_n = (int(self.scaler.n_features_in_)
                        if self.scaler is not None and hasattr(self.scaler, "n_features_in_")
                        else None)
            if scaler_n is not None and scaler_n != num_features:
                print(f"[WARN] num_features={num_features} conflicts with scaler "
                      f"({scaler_n}) — using scaler value to avoid shape error")
                self.num_features = scaler_n
            else:
                self.num_features = num_features

        # ── Load feature names ────────────────────────────────────────
        self.feature_names = []
        if os.path.exists(FEAT_PATH):
            with open(FEAT_PATH, encoding="utf-8") as f:
                self.feature_names = [l.strip() for l in f if l.strip()]

        # ── Load model ────────────────────────────────────────────────
        ckpt = os.path.join(CKPT_DIR, f"{model_name.replace(' ', '_')}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                f"Available: {os.listdir(CKPT_DIR) if os.path.exists(CKPT_DIR) else 'directory missing'}"
            )

        ModelClass = MODEL_CLASSES[model_name]
        self.model = ModelClass(self.num_features)
        self.model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        print(f"[OK] Loaded {model_name} from {ckpt}  (num_features={self.num_features})")

    # ─────────────────────────────────────────────────────────────────

    def classify(self, features: np.ndarray) -> dict:
        """
        Classify a single feature vector of shape (num_features,).
        Returns dict: prediction, label, confidence, prob_attack, prob_benign.
        """
        x = features.copy().astype(np.float32)

        if len(x) != self.num_features:
            raise ValueError(
                f"Feature vector length {len(x)} ≠ model expects {self.num_features}. "
                f"Check that FEATURE_COUNT in traffic_simulator.py = {self.num_features}."
            )

        if self.scaler is not None:
            x = self.scaler.transform(x.reshape(1, -1)).flatten()

        tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            logit = self.model(tensor)
            prob  = torch.sigmoid(logit).item()

        label      = 1 if prob >= self.threshold else 0
        prediction = "ATTACK" if label == 1 else "BENIGN"
        return {
            "prediction":  prediction,
            "label":       label,
            "confidence":  prob if label == 1 else 1 - prob,
            "prob_attack": prob,
            "prob_benign": 1 - prob,
        }

    def classify_batch(self, feature_matrix: np.ndarray) -> list:
        """Classify multiple samples at once."""
        X = feature_matrix.copy().astype(np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        tensor = torch.tensor(X, dtype=torch.float32)
        with torch.inference_mode():
            probs = torch.sigmoid(self.model(tensor)).numpy()
        results = []
        for prob in probs:
            label = 1 if prob >= self.threshold else 0
            results.append({
                "prediction":  "ATTACK" if label == 1 else "BENIGN",
                "label":       label,
                "confidence":  float(prob if label == 1 else 1 - prob),
                "prob_attack": float(prob),
                "prob_benign": float(1 - prob),
            })
        return results

    def set_threshold(self, t: float):
        self.threshold = t
        print(f"[INFO] Threshold set to {t:.2f}")


if __name__ == "__main__":
    import numpy as np
    m = IDS_Model("CNN-LSTM")
    dummy = np.zeros(m.num_features, dtype=np.float32)
    print(f"\nZero vector: {m.classify(dummy)}")
    batch = np.random.randn(5, m.num_features).astype(np.float32)
    print("\nBatch of 5:")
    for r in m.classify_batch(batch):
        print(f"  {r['prediction']:6s}  confidence={r['confidence']:.3f}")
