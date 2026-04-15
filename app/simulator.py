import os
import time
import pandas as pd


class CSVSimulator:
    """Replays a labelled network-traffic CSV as a live stream."""

    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        self.has_labels = "Label" in df.columns
        if self.has_labels:
            self.labels = df["Label"].values
            self.features = df.drop(columns=["Label"])
        else:
            self.labels = None
            self.features = df

        self.total_rows = len(df)
        self.current_idx = 0

    def get_features(self):
        """Returns feature column names."""
        return self.features.columns.tolist()

    def generate_flows(self, batch_size=1, speed=1.0):
        """Generator that yields (feature_row, true_label) one at a time."""
        while self.current_idx < self.total_rows:
            end_idx = min(self.current_idx + batch_size, self.total_rows)
            batch_features = self.features.iloc[self.current_idx:end_idx].values.tolist()
            batch_labels = self.labels[self.current_idx:end_idx].tolist() if self.has_labels else None

            for i, row in enumerate(batch_features):
                true_label = batch_labels[i] if batch_labels else None
                yield row, true_label
                time.sleep(0.1 / speed)

            self.current_idx = end_idx

    def reset(self):
        self.current_idx = 0
