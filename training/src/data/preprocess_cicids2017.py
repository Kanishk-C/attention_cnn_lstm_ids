"""
src/data/preprocess_cicids2017.py

Loads all 8 CIC-IDS2017 CSV files, cleans them, encodes labels,
normalises features, and saves train/test splits as .npy files.

Run once:
    python src/data/preprocess_cicids2017.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DIR = os.path.join("data", "raw", "cicids2017")
OUT_DIR = os.path.join("data", "processed")

CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp",
    "Src IP", "Dst IP", "src_ip", "dst_ip",
]


def _read_csv(path):
    """Read CSV with UTF-8 fallback to latin-1 for CIC files."""
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"  [WARN] UTF-8 failed, retrying latin-1: {os.path.basename(path)}")
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def load_and_merge():
    dfs = []
    for fname in CSV_FILES:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] Missing, skipping: {fname}")
            continue
        print(f"[INFO] Loading {fname}")
        df = _read_csv(path)
        df.columns = df.columns.str.strip()   # remove leading/trailing spaces
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}\n"
            "Download from: https://www.unb.ca/cic/datasets/ids-2017.html"
        )
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined: {combined.shape[0]:,} rows, {combined.shape[1]} columns")
    return combined


def clean(df):
    drop = [c for c in df.columns if c.strip() in DROP_COLS]
    if drop:
        df = df.drop(columns=drop)
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"[INFO] Removed {before - len(df):,} rows with NaN/inf")
    print(f"[INFO] Clean shape: {df.shape}")
    return df


def encode_labels(df):
    # Find label column case-insensitively
    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"), None
    )
    if label_col is None:
        raise ValueError(f"No 'Label' column found. Columns: {list(df.columns)[:8]}")

    df = df.copy()
    df[label_col] = (df[label_col].astype(str).str.strip() != "BENIGN").astype(int)
    y = df[label_col].values
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])

    print(f"[INFO] Features : {X.shape[1]}")
    print(f"[INFO] BENIGN   : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"[INFO] ATTACK   : {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    return X, y


def split_and_scale(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler


def save(X_train, X_test, y_train, y_test, feature_names, scaler):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test.astype(np.float32))
    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    print(f"[INFO] Saved to {OUT_DIR}/")


def main():
    print("=" * 55)
    print("  CIC-IDS2017 Preprocessing")
    print("=" * 55)
    df = load_and_merge()
    df = clean(df)
    X, y = encode_labels(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    save(X_train, X_test, y_train, y_test, X.columns.tolist(), scaler)
    print()
    print("[DONE]")
    print(f"  Train : {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"  Test  : {X_test.shape[0]:,} rows")
    print()
    print("Next:  python run_all.py --epochs 3 --model CNN")


if __name__ == "__main__":
    main()
