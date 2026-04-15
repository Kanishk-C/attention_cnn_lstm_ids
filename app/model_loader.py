import os
import sys
import warnings
import numpy as np
import torch
import joblib
import pandas as pd
from sklearn.metrics import f1_score

# Add the training/ directory so src.models can be imported
TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
if TRAINING_DIR not in sys.path:
    sys.path.append(TRAINING_DIR)

from src.models.attention_cnn_lstm import AttentionCNNLSTM
from src.models.cnn_baseline import CNNBaseline
from src.models.lstm_baseline import LSTMBaseline
from src.models.cnn_lstm import CNNLSTM

_MODEL_MAP = {
    "Attention-CNN-LSTM": AttentionCNNLSTM,
    "CNN": CNNBaseline,
    "LSTM": LSTMBaseline,
    "CNN-LSTM": CNNLSTM,
}


class ModelLoader:
    def __init__(self, model_name="Attention-CNN-LSTM", threshold=0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.scaler = None
        self.model = None
        self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.load_scaler()
        self.load_model()

    def load_scaler(self):
        scaler_path = os.path.join(self.ROOT_DIR, "data", "processed", "scaler.pkl")
        try:
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Failed to load scaler from {scaler_path}: {e}")
            raise

    def load_model(self):
        ckpt_path = os.path.join(self.ROOT_DIR, "experiments", "checkpoints", f"{self.model_name}.pt")
        num_features = len(self.scaler.mean_) if self.scaler else 78

        if self.model_name not in _MODEL_MAP:
            raise ValueError(f"Unknown model name: {self.model_name}")
        self.model = _MODEL_MAP[self.model_name](num_features)

        try:
            # Load to CPU for compatibility across all deployment environments
            self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            self.model.eval()
        except Exception as e:
            print(f"Failed to load weights from {ckpt_path}: {e}")
            raise

    def _scale(self, X):
        """Apply StandardScaler, suppressing sklearn feature-name warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.scaler.transform(X)

    def classify(self, raw_features):
        """Classify a single flow. Accepts a list, dict, or DataFrame row.
        Returns (predicted_class, confidence, attack_probability).
        """
        if isinstance(raw_features, dict):
            X = pd.DataFrame([raw_features]).values
        elif isinstance(raw_features, pd.DataFrame):
            X = raw_features.values
        else:
            X = np.array(raw_features)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        X_tensor = torch.tensor(self._scale(X), dtype=torch.float32)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(X_tensor)).item()

        is_attack = prob >= self.threshold
        confidence = prob if is_attack else (1.0 - prob)
        return ("ATTACK" if is_attack else "BENIGN"), confidence, prob

    def classify_batch(self, raw_features_list):
        """Classify a batch of flows (each a list of 78 floats).
        Returns a list of (predicted_class, confidence, attack_probability).
        """
        if not raw_features_list:
            return []

        X = np.array(raw_features_list)
        if X.shape[1] > 78:
            X = X[:, :78]

        X_tensor = torch.tensor(self._scale(X), dtype=torch.float32)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(X_tensor)).squeeze(-1)

        if prob.ndim == 0:
            prob = prob.unsqueeze(0)
        probs = prob.cpu().numpy()

        results = []
        for p in probs:
            is_attack = p >= self.threshold
            confidence = float(p) if is_attack else float(1.0 - p)
            results.append(("ATTACK" if is_attack else "BENIGN", confidence, float(p)))
        return results

    def autotune_threshold(self):
        """Sweep thresholds on sample_traffic.csv and pick the one with the best F1.
        Returns (best_threshold, best_f1).
        """
        sample_path = os.path.join(self.ROOT_DIR, "app", "data", "sample_traffic.csv")
        if not os.path.exists(sample_path):
            return self.threshold, 0.0

        try:
            df = pd.read_csv(sample_path)
            if "Label" not in df.columns:
                return self.threshold, 0.0

            skip = {"Label", "src_ip", "dst_ip", "src_port", "dst_port", "Protocol", "Timestamp"}
            feature_cols = [c for c in df.columns if c not in skip][:78]
            X_tensor = torch.tensor(self._scale(df[feature_cols].values), dtype=torch.float32)

            with torch.no_grad():
                probs = torch.sigmoid(self.model(X_tensor)).squeeze(-1).cpu().numpy()

            y = (df["Label"] == "ATTACK").astype(int).values if df["Label"].dtype == object \
                else df["Label"].astype(int).values

            best_t, best_f1 = 0.5, 0.0
            for t in np.arange(0.01, 0.99, 0.02):
                f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t

            self.threshold = float(best_t)
            return self.threshold, float(best_f1)
        except Exception as e:
            print(f"Auto-tune failed: {e}")
            return self.threshold, 0.0
