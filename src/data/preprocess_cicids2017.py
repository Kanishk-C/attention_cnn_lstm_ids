import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DIR = "data/raw/cicids2017"
OUT_DIR = "data/processed"

CSV_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]

DROP_COLS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
]

def load_and_merge():
    dfs = []
    for file in CSV_FILES:
        path = os.path.join(RAW_DIR, file)
        print(f"[INFO] Loading {file}")
        df = pd.read_csv(path)

        # Fix column names
        df.columns = df.columns.str.strip()

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined shape: {df.shape}")
    return df

def clean_dataframe(df):
    # Drop identifier columns if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Replace inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN
    df.dropna(inplace=True)

    print(f"[INFO] After cleaning: {df.shape}")
    return df

def encode_labels(df):
    # Binary encoding
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    y = df["Label"].values
    X = df.drop(columns=["Label"])

    return X, y

def normalize_and_split(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test

def save_outputs(X_train, X_test, y_train, y_test, feature_names):
    os.makedirs(OUT_DIR, exist_ok=True)

    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    print("[INFO] Saved processed datasets")

def main():
    df = load_and_merge()
    df = clean_dataframe(df)
    X, y = encode_labels(df)
    X_train, X_test, y_train, y_test = normalize_and_split(X, y)
    save_outputs(X_train, X_test, y_train, y_test, X.columns)

    print("[DONE] Preprocessing complete")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")

if __name__ == "__main__":
    main()

