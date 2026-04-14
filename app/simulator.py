import pandas as pd
import time
import os

class CSVSimulator:
    def __init__(self, file_path):
        """
        Initializes the simulator with a given CSV file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at {file_path}")
            
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        
        # Determine if there's a ground truth label column
        self.has_labels = "Label" in self.df.columns
        if self.has_labels:
            self.labels = self.df["Label"].values
            self.features = self.df.drop(columns=["Label"])
        else:
            self.labels = None
            self.features = self.df
            
        self.total_rows = len(self.df)
        self.current_idx = 0

    def get_features(self):
        """Returns the list of feature column names."""
        return self.features.columns.tolist()

    def generate_flows(self, batch_size=1, speed=1.0):
        """
        A generator that yields rows sequentially, simulating live network traffic.
        """
        while self.current_idx < self.total_rows:
            # Slicing the dataframe
            end_idx = min(self.current_idx + batch_size, self.total_rows)
            batch_features = self.features.iloc[self.current_idx:end_idx].values.tolist()
            
            batch_labels = None
            if self.has_labels:
                batch_labels = self.labels[self.current_idx:end_idx].tolist()
            
            # Since we want to emulate streaming, we yield one entry at a time if batch=1
            # For UI responsiveness we usually pull batches of 5-10
            for i in range(len(batch_features)):
                row = batch_features[i]
                true_label = batch_labels[i] if batch_labels else None
                
                yield row, true_label
                time.sleep(0.1 / speed) # Adjusting playback speed

            self.current_idx = end_idx
            
    def reset(self):
        self.current_idx = 0
