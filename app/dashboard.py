import os
import sys
import time
import csv
import pandas as pd
import streamlit as st
import plotly.express as px
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
    
    /* Ensure dataframe content is readable in dark mode */
    [data-testid="stDataFrame"] {
        color: #C9D1D9 !important;
    }
    
    /* Ensure chart text is visible */
    .js-plotly-plot .plotly text {
        fill: #C9D1D9 !important;
    }
</style>


""", unsafe_allow_html=True)

# State Initialization
if "dashboard_metrics" not in st.session_state:
    st.session_state["dashboard_metrics"] = {
        "packets_processed": 0,
        "benign_count": 0,
        "malicious_count": 0,
        "log_queue": []
    }
if "simulator" not in st.session_state:
    st.session_state["simulator"] = None
if "file_position" not in st.session_state:
    st.session_state["file_position"] = 0


@st.cache_resource
def load_ids_model(model_name="Attention-CNN-LSTM", threshold=0.5):
    return ModelLoader(model_name=model_name, threshold=threshold)

# Sidebar Confguration
st.sidebar.title("🛡️ IDS Settings")
st.sidebar.markdown("Configure real-time monitoring and model performance.")

# Reset Logic
if st.sidebar.button("🗑️ Reset Demo / Clear Captures", type="primary", use_container_width=True):
    st.session_state["dashboard_metrics"] = {
        "packets_processed": 0,
        "benign_count": 0,
        "malicious_count": 0,
        "log_queue": []
    }
    captures_dir = os.path.join(ROOT_DIR, "app", "data", "captures")
    if os.path.exists(captures_dir):
        for filename in os.listdir(captures_dir):
            file_path = os.path.join(captures_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception:
                pass
    st.sidebar.success("Environment Reset.")
    time.sleep(1)
    st.rerun()

mode = st.sidebar.radio("Traffic Mode", ["Live Capture (NFStream)", "Simulation (CSV)"])


st.sidebar.subheader("Model Configuration")
selected_model = st.sidebar.selectbox("Select Model Architecture", 
                                    ["Attention-CNN-LSTM", "CNN", "LSTM", "CNN-LSTM"])

if "model_threshold" not in st.session_state:
    st.session_state["model_threshold"] = 0.50

threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, st.session_state["model_threshold"], 0.01)

# Initialize chosen model
model_loader = load_ids_model(model_name=selected_model, threshold=threshold)
st.sidebar.success(f"{selected_model} Loaded.")

if st.sidebar.button("⚙️ Auto-Tune Threshold", help="Find best threshold using sample test data to maximize F1 Score"):
    with st.spinner("Finding optimal threshold..."):
        best_t, best_f1 = model_loader.autotune_threshold()
        if best_f1 > 0:
            st.session_state["model_threshold"] = round(best_t, 2)
            st.sidebar.success(f"Optimized Threshold: {best_t:.2f} (F1: {best_f1:.2f})")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("Could not run autotune. Sample data might be missing.")

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
        temp_path = os.path.join(ROOT_DIR, "app", "data", "captures", ".temp_upload.csv")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["simulator"] = CSVSimulator(temp_path)
        
    simulator_speed = st.sidebar.slider("Playback Speed (x)", 0.1, 5.0, 1.0)
else:
    st.sidebar.info("Running in Live Mode. Ensure `sudo python app/capture_agent.py` is running in the background.")

# Tabs Setup
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
        # Path Resolution (must match capture_agent.py priority)
        if os.path.exists("/dev/shm") and os.access("/dev/shm", os.R_OK):
            QUEUE_URL = "/dev/shm/ids_traffic_queue.csv"
        else:
            QUEUE_URL = os.path.join(ROOT_DIR, "app", "data", "captures", ".traffic_queue.csv")
        
        HISTORY_FILE = os.path.join(ROOT_DIR, "app", "data", "captures", "attack_history.csv")



        SIMULATOR = st.session_state.get("simulator")
        
        sim_generator = None
        if mode == "Simulation (CSV)" and SIMULATOR is not None:
            sim_generator = SIMULATOR.generate_flows(speed=simulator_speed)
            
        metrics = st.session_state["dashboard_metrics"]
        
        try:
            while True:
                new_flows = []

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
                else:  # Live Mode
                    if os.path.exists(QUEUE_URL) and os.path.getsize(QUEUE_URL) > 0:
                        try:
                            file_size = os.path.getsize(QUEUE_URL)
                            if st.session_state["file_position"] > file_size:
                                st.session_state["file_position"] = 0  # file rolled over

                            with open(QUEUE_URL, 'r') as f:
                                f.seek(st.session_state["file_position"])
                                lines = f.readlines()

                                if lines:
                                    from io import StringIO
                                    reader = csv.reader(StringIO("".join(lines)))
                                    for row in reader:
                                        if len(row) < 82 or row[0] == "src_ip":
                                            continue
                                        new_flows.append({
                                            "src_ip": row[0],
                                            "dst_ip": row[1],
                                            "port": int(row[3]) if row[3].isdigit() else 0,
                                            "features": [float(v) for v in row[4:82]]
                                        })

                                st.session_state["file_position"] = f.tell()
                        except Exception:
                            pass


                
                # Process flows in batches
                BATCH_SIZE = 100
                for i in range(0, len(new_flows), BATCH_SIZE):
                    batch_flows = new_flows[i:i+BATCH_SIZE]
                    
                    batch_features = []
                    for flow in batch_flows:
                        if isinstance(flow, dict):
                            features = flow["features"]
                        else:
                            features = flow
                        
                        if len(features) < 78:
                            features += [0.0] * (78 - len(features))
                        batch_features.append(features[:78])
                    
                    batch_results = model_loader.classify_batch(batch_features)
                    
                    for flow, (pred_class, conf, prob) in zip(batch_flows, batch_results):
                        # Handle both Simulation (list) and Live (dict) formats
                        if isinstance(flow, dict):
                            src_ip = flow["src_ip"]
                            dst_ip = flow["dst_ip"]
                            port = flow["port"]
                        else:
                            src_ip = "Simulation"
                            dst_ip = "Dataset"
                            port = flow[0] if len(flow) > 0 else "N/A"
                            
                        metrics["packets_processed"] += 1
                        if pred_class == "ATTACK":
                            metrics["malicious_count"] += 1
                            # Persistent Attack Logging (SSD)
                            try:
                                file_exists = os.path.exists(HISTORY_FILE)
                                with open(HISTORY_FILE, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    if not file_exists or os.path.getsize(HISTORY_FILE) == 0:
                                        writer.writerow(["Time", "Src IP", "Dest IP", "Port", "Confidence", "P(Attack)"])
                                    writer.writerow([
                                        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        src_ip, dst_ip, port, round(conf * 100, 2), round(prob, 4)
                                    ])
                            except Exception:
                                pass
                        else:
                            metrics["benign_count"] += 1
    
                        metrics["log_queue"].insert(0, {
                            "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                            "Src IP": src_ip,
                            "Dest IP": dst_ip,
                            "Port": port,
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
                chart_placeholder.plotly_chart(fig, key=f"donut_{int(time.time()*1000)}")

                
                if mode == "Live Capture (NFStream)":
                    time.sleep(0.3)
                else:
                    time.sleep(0.01)

                    
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
        st.plotly_chart(fig_bar, key="bar_chart")
    else:
        st.warning("Training results not found. Make sure you have trained the models and `logs/results.csv` exists.")



with tab3:
    st.header("Training History & Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
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

    st.divider()
    st.header("📜 Persistent Attack History")
    st.markdown("This log is saved to physical storage and persists across sessions.")
    history_path = os.path.join(ROOT_DIR, "app", "data", "captures", "attack_history.csv")
    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        df_hist = pd.read_csv(history_path)
        st.dataframe(df_hist.sort_values(by="Time", ascending=False), width='stretch')
        if st.button("🧹 Clear History File"):
             os.remove(history_path)
             st.rerun()
    else:
        st.info("No attacks logged yet.")

