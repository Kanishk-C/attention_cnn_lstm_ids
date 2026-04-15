import os
import sys
import torch
import joblib
import pandas as pd

# Important: Add the /training directory to the sys.path so we can import src.models
TRAINING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
if TRAINING_DIR not in sys.path:
    sys.path.append(TRAINING_DIR)

from src.models.attention_cnn_lstm import AttentionCNNLSTM
from src.models.cnn_baseline import CNNBaseline
from src.models.lstm_baseline import LSTMBaseline
from src.models.cnn_lstm import CNNLSTM

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
        
        # Determine number of features from scaler
        num_features = len(self.scaler.mean_) if self.scaler else 78
        
        # Instantiate correct model class based on name
        if self.model_name == "Attention-CNN-LSTM":
            self.model = AttentionCNNLSTM(num_features)
        elif self.model_name == "CNN":
            self.model = CNNBaseline(num_features)
        elif self.model_name == "LSTM":
            self.model = LSTMBaseline(num_features)
        elif self.model_name == "CNN-LSTM":
            self.model = CNNLSTM(num_features)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        try:
            # Always load to CPU to avoid strict GPU requirements on deployment/dashboard
            self.model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
            self.model.eval()
        except Exception as e:
            print(f"Failed to load weights from {ckpt_path}: {e}")
            raise

    def classify(self, raw_features):
        """
        Takes raw features (as a dict, list, or pandas DataFrame row).
        Outputs: (Predicted_Class (0=BENIGN, 1=ATTACK), Confidence)
        """
        if isinstance(raw_features, dict):
            # Convert dictionary to a 2D 1-row DataFrame
            df = pd.DataFrame([raw_features])
            # Ensure it is in the exact order of the scaler
            X = df.values
        elif isinstance(raw_features, pd.DataFrame):
            X = raw_features.values
        else:
            import numpy as np
            X = np.array(raw_features)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        # Preprocess features using the fitted scaler
        import warnings
        with warnings.catch_warnings():
            # Suppress standardizing warnings from sklearn regarding missing feature names
            warnings.simplefilter("ignore")
            X_scaled = self.scaler.transform(X)

        # Convert to PyTorch Tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Run inference
        with torch.no_grad():
            logit = self.model(X_tensor)
            prob = torch.sigmoid(logit).item()
        
        # Determine prediction
        is_attack = prob >= self.threshold
        confidence = prob if is_attack else (1.0 - prob)
        
        predicted_class = "ATTACK" if is_attack else "BENIGN"
        return predicted_class, confidence, prob

    def classify_batch(self, raw_features_list):
        """
        Takes a list of raw features (each a list of 78 floats).
        Outputs a list of: (Predicted_Class, Confidence, Prob)
        """
        if not raw_features_list:
            return []

        import numpy as np
        X = np.array(raw_features_list)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # If scaling fails due to columns mismatch, ensure 78 cols
            if X.shape[1] > 78:
                X = X[:, :78]
            X_scaled = self.scaler.transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logit = self.model(X_tensor)
            prob = torch.sigmoid(logit).squeeze(-1)
        
        if prob.ndim == 0:
            prob = prob.unsqueeze(0)
            
        probs = prob.cpu().numpy()
        
        results = []
        for p in probs:
            is_attack = p >= self.threshold
            confidence = float(p) if is_attack else float(1.0 - p)
            predicted_class = "ATTACK" if is_attack else "BENIGN"
            results.append((predicted_class, confidence, float(p)))
            
        return results

    def autotune_threshold(self):
        """Finds the optimal threshold to maximize F1 score on sample data."""
        sample_path = os.path.join(self.ROOT_DIR, "app", "data", "sample_traffic.csv")
        if not os.path.exists(sample_path):
            return self.threshold, 0.0
            
        import pandas as pd
        from sklearn.metrics import f1_score
        import numpy as np
        
        try:
            df = pd.read_csv(sample_path)
            if "Label" not in df.columns:
                return self.threshold, 0.0
                
            # Use only feature columns up to 78
            feature_cols = [c for c in df.columns if c not in ["Label", "src_ip", "dst_ip", "src_port", "dst_port", "Protocol", "Timestamp"]]
            if len(feature_cols) > 78:
                 feature_cols = feature_cols[:78]
            X = df[feature_cols].values
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_scaled = self.scaler.transform(X)
                
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                logit = self.model(X_tensor)
                probs = torch.sigmoid(logit).squeeze(-1).cpu().numpy()
                
            best_t = 0.5
            best_f1 = 0.0
            
            if df["Label"].dtype == object:
                y_numeric = (df["Label"] == "ATTACK").astype(int)
            else:
                y_numeric = df["Label"].astype(int).values
                
            # Test thresholds
            for t in np.arange(0.01, 0.99, 0.02):
                preds = (probs >= t).astype(int)
                f1 = f1_score(y_numeric, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
                    
            self.threshold = float(best_t)
            return self.threshold, float(best_f1)
        except Exception as e:
            print(f"Auto-tune failed: {e}")
            return self.threshold, 0.0

