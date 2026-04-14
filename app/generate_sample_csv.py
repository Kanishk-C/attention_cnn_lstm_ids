# Generate Sample CSV
import os
import numpy as np
import pandas as pd

def generate_sample_csv():
    print("Loading test dataset arrays...")
    x_test_path = os.path.join("data", "processed", "X_test.npy")
    y_test_path = os.path.join("data", "processed", "y_test.npy")
    features_path = os.path.join("data", "processed", "feature_names.txt")

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print("Error: Processed numpy arrays not found. Make sure to run the prepocessor first.")
        return

    # Load only the first 500 records to keep the CSV lightweight
    X_sample = np.load(x_test_path)[:500]
    y_sample = np.load(y_test_path)[:500]

    with open(features_path, "r") as f:
        feature_names = [line.strip() for line in f.readlines() if line.strip()]

    # Create DataFrame
    df = pd.DataFrame(X_sample, columns=feature_names)
    df["Label"] = y_sample

    output_dir = os.path.join("app", "data")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "sample_traffic.csv")
    
    df.to_csv(out_file, index=False)
    print(f"Successfully generated {out_file} with {len(df)} rows.")
    print("You can use this file as a template for faculty testing.")

if __name__ == "__main__":
    generate_sample_csv()
