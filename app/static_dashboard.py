"""
app.py  —  Streamlit dashboard for the IDS project.

Run locally:      streamlit run app.py
Run on Colab:     Use Cell 10 in IDS_Colab_Notebook.ipynb (needs ngrok)

set_page_config MUST be the first Streamlit call.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import joblib

from src.models.cnn_baseline       import CNNBaseline
from src.models.lstm_baseline      import LSTMBaseline
from src.models.cnn_lstm           import CNNLSTM
from src.models.attention_cnn_lstm import AttentionCNNLSTM
from src.evaluation.plots import (
    plot_confusion_matrix, plot_loss_curves,
    plot_comparison_bar, plot_radar,
)

st.set_page_config(
    page_title="IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
)

CKPT_DIR = os.path.join("experiments", "checkpoints")
LOG_DIR  = "logs"
DATA_DIR = os.path.join("data", "processed")

MODELS = {
    "CNN":                CNNBaseline,
    "LSTM":               LSTMBaseline,
    "CNN-LSTM":           CNNLSTM,
    "Attention-CNN-LSTM": AttentionCNNLSTM,
}


@st.cache_resource
def get_data():
    X = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "feature_names.txt"), encoding="utf-8") as f:
        feats = [l.strip() for l in f if l.strip()]
    return X, y, feats


@st.cache_resource
def get_scaler():
    p = os.path.join(DATA_DIR, "scaler.pkl")
    return joblib.load(p) if os.path.exists(p) else None


@st.cache_resource
def get_model(name, nf):
    p = os.path.join(CKPT_DIR, name.replace(" ", "_") + ".pt")
    if not os.path.exists(p):
        return None
    m = MODELS[name](nf)
    # map_location="cpu" allows GPU-trained checkpoints to run on CPU
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m


def has_results():
    return os.path.exists(os.path.join(LOG_DIR, "results.csv"))

def has_data():
    return os.path.exists(os.path.join(DATA_DIR, "X_test.npy"))


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🛡️ IDS Dashboard")
st.sidebar.markdown("Network Intrusion Detection · CIC-IDS2017")
page = st.sidebar.radio("Navigate",
    ["Overview", "Model Results", "Live Prediction", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("PyTorch · Streamlit · Attention-CNN-LSTM")


# ════════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Network Intrusion Detection System")
    st.markdown(
        "Classifies network traffic as **BENIGN** or **ATTACK** using "
        "four deep learning architectures trained on CIC-IDS2017."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset",   "CIC-IDS2017")
    c2.metric("Task",      "Binary classification")
    c3.metric("Models",    "4 architectures")
    st.markdown("---")
    st.markdown("""
    | Model | Key idea |
    |---|---|
    | **CNN** | 1D convolutions detect local feature patterns |
    | **LSTM** | Treats 76 features as a sequence |
    | **CNN-LSTM** | CNN extracts features, LSTM models their sequence |
    | **Attention-CNN-LSTM** | Adds self-attention to focus on informative features |

    **Dataset:** ~2.8 million network flow records across 8 CSV files.
    76 numerical features per row. 80/20 stratified train/test split.
    """)


# ════════════════════════════════════════════════════════════════
# MODEL RESULTS
# ════════════════════════════════════════════════════════════════
elif page == "Model Results":
    st.title("Model Results")

    if not has_results():
        st.warning("No results yet. Train the models first:")
        st.code("python run_all.py", language="bash")
        st.stop()

    df = pd.read_csv(os.path.join(LOG_DIR, "results.csv"))

    st.subheader("Performance summary")
    st.dataframe(
        df.style
          .highlight_max(subset=["Accuracy","F1","MCC"], color="#d4f0c0")
          .format({"Accuracy":"{:.4f}","F1":"{:.4f}","MCC":"{:.4f}"}),
        use_container_width=True,
    )

    st.subheader("Metric comparison")
    bar = os.path.join(LOG_DIR, "comparison_bar.png")
    if os.path.exists(bar):
        st.image(bar, use_container_width=True)

    st.subheader("Training loss curves")
    hist = os.path.join(LOG_DIR, "histories.json")
    lc   = os.path.join(LOG_DIR, "loss_curves.png")
    if os.path.exists(lc):
        st.image(lc, use_container_width=True)
    elif os.path.exists(hist):
        with open(hist, encoding="utf-8") as f:
            st.pyplot(plot_loss_curves(json.load(f)))

    st.subheader("Confusion matrices")
    cols = st.columns(2)
    for i, name in enumerate(MODELS):
        p = os.path.join(LOG_DIR, f"cm_{name.replace(' ','_')}.png")
        with cols[i % 2]:
            if os.path.exists(p):
                st.image(p, caption=name, use_container_width=True)
            else:
                st.info(f"No confusion matrix for {name} yet.")

    st.subheader("Performance radar")
    rad = os.path.join(LOG_DIR, "radar.png")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if os.path.exists(rad):
            st.image(rad, use_container_width=True)
        else:
            rd = {r["Model"]: {"accuracy":r["Accuracy"],"f1":r["F1"],"mcc":r["MCC"]}
                  for _, r in df.iterrows()}
            st.pyplot(plot_radar(rd))


# ════════════════════════════════════════════════════════════════
# LIVE PREDICTION
# ════════════════════════════════════════════════════════════════
elif page == "Live Prediction":
    st.title("Live Prediction")

    if not has_data():
        st.warning("Processed data not found. Run preprocessing first.")
        st.code("python src/data/preprocess_cicids2017.py", language="bash")
        st.stop()

    X_test, y_test, feats = get_data()
    nf = X_test.shape[1]

    sel   = st.selectbox("Model", list(MODELS.keys()))
    model = get_model(sel, nf)

    if model is None:
        st.warning(f"No checkpoint for **{sel}**. Run `python run_all.py` first.")
        st.stop()

    st.success(f"Loaded: **{sel}** ({nf} features)")
    mode = st.radio("Input", ["Random test samples", "Upload CSV"])

    if mode == "Random test samples":
        n = st.slider("Samples", 1, 100, 20)
        if st.button("Predict", type="primary"):
            idx   = np.random.choice(len(X_test), n, replace=False)
            Xs    = torch.tensor(X_test[idx], dtype=torch.float32)
            y_sel = y_test[idx].astype(int)
            with torch.inference_mode():
                probs = torch.sigmoid(model(Xs)).numpy()
            preds = (probs > 0.5).astype(int)

            out = pd.DataFrame({
                "Sample #":   idx,
                "True":       ["ATTACK" if y else "BENIGN" for y in y_sel],
                "Predicted":  ["ATTACK" if p else "BENIGN" for p in preds],
                "Confidence": [f"{max(p,1-p)*100:.1f}%" for p in probs],
                "Correct":    ["✓" if p==y else "✗" for p,y in zip(preds,y_sel)],
            })
            st.dataframe(out, use_container_width=True)

            a, b, c = st.columns(3)
            a.metric("Batch accuracy", f"{(preds==y_sel).mean()*100:.1f}%")
            b.metric("Correct",  f"{int((preds==y_sel).sum())}/{n}")
            c.metric("Attacks detected",
                     f"{int(preds.sum())} / {int(y_sel.sum())} real")

    else:
        up = st.file_uploader("Upload CSV (same 76 features, no Label column)", type=["csv"])
        if up:
            scaler = get_scaler()
            df_up  = pd.read_csv(up, low_memory=False)
            df_up.columns = df_up.columns.str.strip()
            miss = [f for f in feats if f not in df_up.columns]
            if miss:
                st.error(f"Missing {len(miss)} columns: {miss[:5]}...")
                st.stop()
            X_up = df_up[feats].select_dtypes(include=[np.number]).values
            if scaler:
                X_up = scaler.transform(X_up)
            Xt = torch.tensor(X_up.astype(np.float32))
            with torch.inference_mode():
                probs = torch.sigmoid(model(Xt)).numpy()
            preds = (probs > 0.5).astype(int)
            df_up["Prediction"] = ["ATTACK" if p else "BENIGN" for p in preds]
            df_up["Confidence"] = [f"{max(p,1-p)*100:.1f}%" for p in probs]
            st.dataframe(df_up[["Prediction","Confidence"]].head(100),
                         use_container_width=True)
            st.info(f"Classified {len(preds)} records — "
                    f"{int(preds.sum())} flagged as ATTACK ({preds.mean()*100:.1f}%)")


# ════════════════════════════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About")
    st.markdown("""
    ### Reference paper
    > *"Deep learning for network security: an Attention-CNN-LSTM model
    for accurate intrusion detection"* — Alashjaee, Scientific Reports (2025)
    > https://doi.org/10.1038/s41598-025-07706-y

    This implementation extends the original work to **CIC-IDS2017**
    and adds a fully interactive Streamlit dashboard.

    ### Run commands
    ```bash
    pip install -r requirements.txt
    python src/data/preprocess_cicids2017.py
    python run_all.py --epochs 3 --model CNN   # quick test
    python run_all.py                           # full training
    streamlit run app.py                        # open dashboard
    ```
    ### Stack
    Python · PyTorch · scikit-learn · Streamlit · matplotlib · pandas · joblib
    """)
