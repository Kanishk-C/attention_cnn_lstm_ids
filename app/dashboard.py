import os
import sys
import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Setup base paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from app.model_loader import ModelLoader
from app.simulator import CSVSimulator

st.set_page_config(
    page_title="Attention-CNN-LSTM IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Cybersecurity theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    
    /* Metrics customization */
    [data-testid="stMetricValue"] {
        color: #58A6FF;
    }
    
    /* Malicious highlight logic (done conditionally in python but keeping classes clean) */
    .attack-text {
        color: #F85149;
        font-weight: bold;
    }
    .benign-text {
        color: #3FB950;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─── STATE INITIALIZATION ──────────────────────────────────────────────────
if "dashboard_metrics" not in st.session_state:
    st.session_state["dashboard_metrics"] = {
        "packets_processed": 0,
        "benign_count": 0,
        "malicious_count": 0,
        "log_queue": []
    }
if "simulator" not in st.session_state:
    st.session_state["simulator"] = None

@st.cache_resource
def load_ids_model(model_name="Attention-CNN-LSTM", threshold=0.5):
    return ModelLoader(model_name=model_name, threshold=threshold)

# ─── SIDEBAR CONFIGURATION ──────────────────────────────────────────────────
st.sidebar.title("🛡️ IDS Settings")
st.sidebar.markdown("Configure real-time monitoring and model performance.")

mode = st.sidebar.radio("Traffic Mode", ["Live Capture (NFStream)", "Simulation (CSV)"])

st.sidebar.subheader("Model Configuration")
selected_model = st.sidebar.selectbox("Select Model Architecture", 
                                    ["Attention-CNN-LSTM", "CNN", "LSTM", "CNN-LSTM"])
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.01)

# Initialize chosen model
model_loader = load_ids_model(model_name=selected_model, threshold=threshold)
st.sidebar.success(f"{selected_model} Loaded.")

simulator_speed = 1.0
uploaded_file = None
if mode == "Simulation (CSV)":
    st.sidebar.subheader("Simulation Controls")
    uploaded_file = st.sidebar.file_uploader("Upload Network Traffic CSV", type="csv")
    if st.sidebar.button("Use Sample Traffic"):
        sample_path = os.path.join(ROOT_DIR, "app", "data", "sample_traffic.csv")
        if os.path.exists(sample_path):
            st.session_state["simulator"] = CSVSimulator(sample_path)
            st.sidebar.success("Loaded Sample Traffic")
        else:
            st.sidebar.error("Sample CSV not found. Run generate_sample_csv.py")
            
    if uploaded_file is not None:
        # Save temp and load
        temp_path = os.path.join(ROOT_DIR, "app", ".temp_upload.csv")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["simulator"] = CSVSimulator(temp_path)
        
    simulator_speed = st.sidebar.slider("Playback Speed (x)", 0.1, 5.0, 1.0)
else:
    st.sidebar.info("Running in Live Mode. Ensure `sudo python app/capture_agent.py` is running in the background.")

# ─── UI TABS ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🚀 Real-Time Dashboard", "📊 Model Comparison", "📈 Visualizations"])

with tab1:
    st.title("Network Intrusion Detection Dashboard")
    st.markdown("Monitoring incoming network flows for malicious activity using PyTorch Deep Learning models.")
    
    col1, col2, col3, col4 = st.columns(4)
    m_total = col1.empty()
    m_benign = col2.empty()
    m_attack = col3.empty()
    m_rate = col4.empty()
    
    st.subheader("Live Traffic Stream")
    table_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    start_btn = st.button("▶ Start Monitoring")
    
    if start_btn:
        QUEUE_URL = os.path.join(ROOT_DIR, "app", ".traffic_queue.json")
        SIMULATOR = st.session_state.get("simulator")
        
        sim_generator = None
        if mode == "Simulation (CSV)" and SIMULATOR is not None:
            sim_generator = SIMULATOR.generate_flows(speed=simulator_speed)
            
        metrics = st.session_state["dashboard_metrics"]
        
        # Polling loop
        try:
            while True:
                new_flows = []
                
                # Fetch data based on mode
                if mode == "Simulation (CSV)":
                    if sim_generator:
                        try:
                            # Fetch one row
                            raw_feat, truth = next(sim_generator)
                            new_flows.append(raw_feat)
                        except StopIteration:
                            st.info("Simulation completed!")
                            break
                    else:
                        st.warning("Please upload a CSV or load the sample traffic in the sidebar first.")
                        break
                else: # Live Mode
                    if os.path.exists(QUEUE_URL):
                        try:
                            with open(QUEUE_URL, "r") as f:
                                live_data = json.load(f)
                            if live_data:
                                new_flows = [live_data[-1]] # Grab latest
                                # Truncate file so we don't re-read
                                with open(QUEUE_URL, "w") as f:
                                    json.dump([], f)
                        except Exception:
                            pass
                
                # Process flows
                for flow in new_flows:
                    # Pad to 78 features if nfstream truncated it
                    if len(flow) < 78:
                        flow += [0.0] * (78 - len(flow))
                        
                    pred_class, conf, prob = model_loader.classify(flow[:78])
                    
                    metrics["packets_processed"] += 1
                    if pred_class == "ATTACK":
                        metrics["malicious_count"] += 1
                    else:
                        metrics["benign_count"] += 1
                        
                    # Maintain log queue size limited to 20
                    metrics["log_queue"].insert(0, {
                        "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "Prediction": f"🚨 {pred_class}" if pred_class == "ATTACK" else f"✅ {pred_class}",
                        "Confidence (%)": round(conf * 100, 2),
                        "P(Attack)": round(prob, 4)
                    })
                    if len(metrics["log_queue"]) > 20:
                        metrics["log_queue"].pop()
                
                # Update UI Dashboard Metrics
                m_total.metric("Total Flows Processed", metrics["packets_processed"])
                m_benign.metric("Normal Traffic (BENIGN)", metrics["benign_count"])
                m_attack.metric("Threat Count (ATTACK)", metrics["malicious_count"])
                
                atk_rate = (metrics["malicious_count"] / max(metrics["packets_processed"], 1)) * 100
                m_rate.metric("Threat Ratio", f"{atk_rate:.1f}%")
                
                # Update Table
                if metrics["log_queue"]:
                    df_log = pd.DataFrame(metrics["log_queue"])
                    table_placeholder.dataframe(df_log, width='stretch')
                
                # Update Chart (Donut)
                fig = px.pie(
                    names=["BENIGN", "ATTACK"],
                    values=[metrics["benign_count"], metrics["malicious_count"]],
                    hole=0.4,
                    color=["BENIGN", "ATTACK"],
                    color_discrete_map={"BENIGN": "#3FB950", "ATTACK": "#F85149"}
                )
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                chart_placeholder.plotly_chart(fig)
                
                # Sleep depending on mode
                if mode == "Live Capture (NFStream)":
                    time.sleep(1.0)
                else:
                    time.sleep(0.01) # fast simulation renders
                    
        except KeyboardInterrupt:
            st.info("Monitoring Stopped.")


with tab2:
    st.header("Comparative Model Performance")
    results_path = os.path.join(ROOT_DIR, "logs", "results.csv")
    
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        st.dataframe(results_df, width='stretch')
        
        st.subheader("Accuracy & F1 Score Comparison")
        fig_bar = px.bar(results_df, x="Model", y=["Accuracy", "F1"], barmode="group",
                         color_discrete_sequence=["#58A6FF", "#238636"])
        st.plotly_chart(fig_bar)
    else:
        st.warning("Training results not found. Make sure you have trained the models and `logs/results.csv` exists.")


with tab3:
    st.header("Training History & Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        # Load from logs
        cm_path = os.path.join(ROOT_DIR, "logs", f"cm_{selected_model.replace(' ', '_')}.png")
        if os.path.exists(cm_path):
            img = Image.open(cm_path)
            st.image(img, caption=f"{selected_model} Confusion Matrix")
        else:
            st.info(f"No CM found for {selected_model}")
            
    with col2:
        st.subheader("Loss Curves")
        loss_path = os.path.join(ROOT_DIR, "logs", "loss_curves.png")
        if os.path.exists(loss_path):
            img = Image.open(loss_path)
            st.image(img, caption="Training Loss Histories")
        else:
            st.info("No global loss curves found.")
            
    st.subheader("Radar Benchmarks")
    radar_path = os.path.join(ROOT_DIR, "logs", "radar.png")
    if os.path.exists(radar_path):
        st.image(Image.open(radar_path))
