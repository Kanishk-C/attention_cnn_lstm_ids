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
