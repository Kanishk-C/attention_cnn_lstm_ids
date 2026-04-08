"""
live_demo.py  —  Real-time IDS Dashboard (v3)
Run from project ROOT:
    streamlit run live_demo/live_demo.py

NEW in v3
─────────────────────────────────────────────────────
1. REAL TRAFFIC MODE  — receives live feature vectors POSTed by
   capture_agent.py running on any device (phone, laptop, VM).
   Uses a shared JSON queue file for zero-dependency IPC.

2. SIMULATED ATTACK DEVICE — sidebar button crafts realistic
   attack vectors and injects them as if from a remote device,
   with a labelled source so you can tell them apart in the log.

3. AUTO-TUNE FIX — samples max 2 000 rows (mmap), no crash.

4. STAR-BUTTON FIX — threshold synced to model.threshold before
   every single prediction, not just on reload.

5. CLEAN TAB LAYOUT — Dashboard | Real Traffic | Attack Device | Help
"""

import os, sys, time, json, random, threading
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from simulator import (
    generate_traffic_sample, ATTACK_PROFILES,
    _attack_vector, FEATURE_COUNT,
)
from model_loader import IDS_Model

try:
    import psutil as _psutil; _PSUTIL = True
except ImportError:
    _psutil = None; _PSUTIL = False

# ── Queue file for real-device traffic ────────────────────────────────────────
# capture_agent.py writes to app/.traffic_queue.json — must match exactly
QUEUE_FILE = _HERE / ".traffic_queue.json"

def _read_queue(max_items=50):
    """Pop up to max_items from the JSON queue. Returns list of dicts."""
    if not QUEUE_FILE.exists():
        return []
    try:
        with open(QUEUE_FILE, "r") as f:
            data = json.load(f)
        items = data[:max_items]
        remainder = data[max_items:]
        with open(QUEUE_FILE, "w") as f:
            json.dump(remainder, f)
        return items
    except Exception:
        return []

def _push_to_queue(sample_dict):
    """Push one sample dict to the queue (used by inject-from-UI)."""
    try:
        existing = []
        if QUEUE_FILE.exists():
            with open(QUEUE_FILE, "r") as f:
                existing = json.load(f)
        existing.append(sample_dict)
        with open(QUEUE_FILE, "w") as f:
            json.dump(existing, f)
    except Exception:
        pass

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS Live Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

@keyframes flash-red {
  0%,100%{background:rgba(220,38,38,0.15);}
  50%    {background:rgba(220,38,38,0.03);}
}
@keyframes pulse-green {
  0%,100%{box-shadow:0 0 0 0 rgba(22,163,74,0.4);}
  50%    {box-shadow:0 0 0 6px rgba(22,163,74,0);}
}

.attack-alert{border:2px solid #DC2626;border-radius:10px;padding:12px 18px;
  animation:flash-red 1.2s ease-in-out 3;color:#7F1D1D;font-weight:600;
  font-size:14px;margin-bottom:10px;background:rgba(220,38,38,0.07);}
.benign-ok{border:1.5px solid #16A34A;background:#F0FDF4;border-radius:10px;
  padding:10px 16px;color:#14532D;font-size:13px;margin-bottom:6px;}

.cm-tp{background:#F0FDF4;border:2px solid #86EFAC;border-radius:10px;padding:14px;text-align:center;}
.cm-tn{background:#EFF6FF;border:2px solid #93C5FD;border-radius:10px;padding:14px;text-align:center;}
.cm-fp{background:#FFF7ED;border:2px solid #FCD34D;border-radius:10px;padding:14px;text-align:center;}
.cm-fn{background:#FEF2F2;border:2px solid #FCA5A5;border-radius:10px;padding:14px;text-align:center;}
.cm-val{font-size:36px;font-weight:700;line-height:1.2;font-family:'JetBrains Mono',monospace;}
.cm-lbl{font-size:12px;font-weight:600;margin-top:4px;}
.cm-sub{font-size:10px;color:#94A3B8;margin-top:2px;}

.device-badge{display:inline-block;padding:2px 8px;border-radius:12px;
  font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace;}
.badge-real{background:#DBEAFE;color:#1E40AF;}
.badge-sim{background:#F3E8FF;color:#6B21A8;}
.badge-inject{background:#FEF3C7;color:#92400E;}

.code-block{background:#0F172A;color:#E2E8F0;border-radius:10px;
  padding:16px 20px;font-family:'JetBrains Mono',monospace;
  font-size:12px;line-height:1.7;overflow-x:auto;margin:8px 0;}
.code-comment{color:#64748B;}
.code-cmd{color:#38BDF8;}
.code-str{color:#A3E635;}

.net-stat-box{border:1px solid rgba(128,128,128,0.2);border-radius:8px;
  padding:8px 14px;font-size:12px;font-family:'JetBrains Mono',monospace;
  background:#F8FAFC;}

.live-dot{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:#16A34A;animation:pulse-green 1.5s infinite;margin-right:6px;}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
_defaults = dict(
    model=None, running=False, log=[], tp=0, tn=0, fp=0, fn=0,
    last_alert=None, model_name="CNN-LSTM", threshold=0.50,
    prev_net=None, auto_threshold=None,
    traffic_mode="Simulated",   # "Simulated" | "Real Device" | "Mixed"
    real_device_count=0,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🛡️ IDS Live Demo")
    st.markdown("Real-time intrusion detection · CIC-IDS2017")
    st.markdown("---")

    model_choice = st.selectbox(
        "Model", ["CNN-LSTM", "CNN", "LSTM", "Attention-CNN-LSTM"]
    )

    # ── Traffic source ─────────────────────────────────────────────────────────
    st.markdown("**Traffic source**")
    traffic_mode = st.radio(
        "Source",
        ["Simulated", "Real Device", "Mixed"],
        help=(
            "Simulated: synthetic vectors from traffic_simulator.py\n"
            "Real Device: reads from capture_agent.py on your phone/laptop\n"
            "Mixed: both at once"
        ),
        label_visibility="collapsed",
    )
    st.session_state.traffic_mode = traffic_mode

    # ── Threshold ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Detection threshold**")
    threshold = st.slider(
        "Threshold", 0.05, 0.95,
        float(st.session_state.threshold), 0.05,
        key="threshold_slider",
        label_visibility="collapsed",
        help="P(Attack) ≥ threshold → flagged ATTACK",
    )
    st.session_state.threshold = threshold

    # ── Lightweight auto-tune (max 2 000 rows, no crash) ──────────────────────
    if st.button("🎯 Auto-tune (max F1)", help="Samples 2 000 rows — fast, no crash"):
        data_dir = _ROOT / "data" / "processed"
        try:
            import joblib, torch
            from model_loader import MODEL_CLASSES as _cls_map
            _cls = _cls_map

            X_mm = np.load(data_dir / "X_test.npy", mmap_mode="r")
            y_mm = np.load(data_dir / "y_test.npy", mmap_mode="r")
            n    = X_mm.shape[0]
            rng  = np.random.default_rng(42)
            idx  = np.sort(rng.choice(n, min(2000, n), replace=False))
            X_s  = joblib.load(data_dir / "scaler.pkl").transform(
                       X_mm[idx].copy().astype(np.float32))
            mdl  = _cls[model_choice](X_s.shape[1])
            mdl.load_state_dict(torch.load(
                _ROOT / "experiments" / "checkpoints" /
                f"{model_choice.replace(' ','_')}.pt",
                map_location="cpu", weights_only=True))
            mdl.eval()

            with torch.inference_mode():
                probs = torch.sigmoid(
                    mdl(torch.tensor(X_s, dtype=torch.float32))
                ).numpy().flatten()

            y_sub = y_mm[idx].copy().astype(int)
            best_t, best_f1 = 0.50, 0.0
            for t in np.linspace(0.05, 0.95, 50):
                p  = (probs >= t).astype(int)
                tp_ = int(((p==1)&(y_sub==1)).sum())
                fp_ = int(((p==1)&(y_sub==0)).sum())
                fn_ = int(((p==0)&(y_sub==1)).sum())
                d   = 2*tp_ + fp_ + fn_
                f1_ = 2*tp_/d if d else 0.0
                if f1_ > best_f1:
                    best_f1, best_t = f1_, float(t)

            st.session_state.auto_threshold = best_t
            st.session_state.threshold      = best_t
            st.success(f"✓ Optimal: **{best_t:.2f}** (F1={best_f1:.4f}, n={len(idx):,})")
        except Exception as e:
            st.warning(f"Auto-tune failed: {e}")

    if st.session_state.auto_threshold is not None:
        at = st.session_state.auto_threshold
        st.info(f"Auto-tuned: **{at:.2f}**")
        if st.button("Apply"):
            st.session_state.threshold = at
            st.rerun()

    # ── Stream controls ────────────────────────────────────────────────────────
    st.markdown("---")
    speed     = st.selectbox("Sample rate",
                             ["Fast (0.5 s)", "Normal (1 s)", "Slow (2 s)"], index=1)
    delay     = {"Fast (0.5 s)": 0.5, "Normal (1 s)": 1.0, "Slow (2 s)": 2.0}[speed]
    attack_mix = st.slider("Simulated attack %", 0, 100, 30, 5)

    st.markdown("---")
    c1, c2 = st.columns(2)
    start_btn = c1.button("▶ Start", type="primary", use_container_width=True)
    stop_btn  = c2.button("⏹ Stop",                 use_container_width=True)
    clear_btn = st.button("🗑 Clear log", use_container_width=True)
    st.caption(f"Feature vector: **{FEATURE_COUNT}** dims")
    if not _PSUTIL:
        st.caption("_psutil not installed — no live net I/O stats_")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD / SYNC MODEL
# ══════════════════════════════════════════════════════════════════════════════
if (st.session_state.model is None or
        st.session_state.model_name != model_choice):
    with st.spinner(f"Loading {model_choice}…"):
        try:
            m = IDS_Model(model_name=model_choice,
                          num_features=None,
                          threshold=st.session_state.threshold)
            st.session_state.model      = m
            st.session_state.model_name = model_choice

            # ── Auto-tune threshold at model load (no manual click needed) ────
            # X_test.npy is ALREADY z-scored by the preprocessor.
            # Pass it directly to the model — do NOT apply scaler.transform() again.
            _data_dir = _ROOT / "data" / "processed"
            try:
                import torch as _torch
                _X_mm = np.load(_data_dir / "X_test.npy", mmap_mode="r")
                _y_mm = np.load(_data_dir / "y_test.npy", mmap_mode="r")
                _rng  = np.random.default_rng(42)
                _idx  = np.sort(_rng.choice(_X_mm.shape[0], min(2000, _X_mm.shape[0]), replace=False))
                _X_s  = _X_mm[_idx].copy().astype(np.float32)   # already scaled ✓
                m._model_eval = m.model
                with _torch.inference_mode():
                    _probs = _torch.sigmoid(
                        m.model(_torch.tensor(_X_s, dtype=_torch.float32))
                    ).numpy().flatten()
                _y_sub = _y_mm[_idx].copy().astype(int)
                _best_t, _best_f1 = 0.50, 0.0
                for _t in np.linspace(0.05, 0.95, 50):
                    _p   = (_probs >= _t).astype(int)
                    _tp_ = int(((_p==1)&(_y_sub==1)).sum())
                    _fp_ = int(((_p==1)&(_y_sub==0)).sum())
                    _fn_ = int(((_p==0)&(_y_sub==1)).sum())
                    _d   = 2*_tp_ + _fp_ + _fn_
                    _f1  = 2*_tp_/_d if _d else 0.0
                    if _f1 > _best_f1:
                        _best_f1, _best_t = _f1, float(_t)
                m.threshold                     = _best_t
                st.session_state.threshold      = _best_t
                st.session_state.auto_threshold = _best_t
            except Exception:
                pass   # silently keep default 0.50 if data not ready

        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()

model = st.session_state.model
# ★ Always sync threshold — this is the star-button fix
model.threshold = st.session_state.threshold

# ── Button handlers ────────────────────────────────────────────────────────────
if start_btn: st.session_state.running = True
if stop_btn:  st.session_state.running = False
if clear_btn:
    for k in ("log","tp","tn","fp","fn","last_alert","prev_net","real_device_count"):
        st.session_state[k] = [] if k in ("log",) else (
            None if k in ("last_alert","prev_net") else 0)

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFY ONE SAMPLE
# ══════════════════════════════════════════════════════════════════════════════
def process_sample(sample: dict, source: str = "sim") -> dict:
    result     = model.classify(sample["features"])
    true_label = sample["true_label"]
    pred_label = result["label"]

    if   pred_label == 1 and true_label == 1: st.session_state.tp += 1
    elif pred_label == 0 and true_label == 0: st.session_state.tn += 1
    elif pred_label == 1 and true_label == 0: st.session_state.fp += 1
    elif pred_label == 0 and true_label == 1: st.session_state.fn += 1

    if result["prediction"] == "ATTACK":
        st.session_state.last_alert = {
            "time":        sample["timestamp"],
            "source_ip":   sample["source_ip"],
            "attack_type": sample["attack_type"],
            "confidence":  result["confidence"],
            "src":         source,
        }

    return {
        "Time":        sample["timestamp"],
        "Source IP":   sample["source_ip"],
        "Port":        sample["dest_port"],
        "Origin":      source,
        "True label":  "ATTACK" if true_label == 1 else "BENIGN",
        "Predicted":   result["prediction"],
        "Confidence":  f"{result['confidence']*100:.1f}%",
        "P(Atk)":      f"{result['prob_attack']*100:.1f}%",
        "✓/✗":        "✓" if pred_label == true_label else "✗",
        "_pred_attack": result["prediction"] == "ATTACK",
        "_correct":     pred_label == true_label,
        "_true_attack": true_label == 1,
    }

# ══════════════════════════════════════════════════════════════════════════════
# INGEST REAL DEVICE PACKETS FROM QUEUE
# ══════════════════════════════════════════════════════════════════════════════
def ingest_real_traffic():
    """Read whatever capture_agent.py has pushed to the queue file."""
    items = _read_queue(max_items=20)
    for item in items:
        try:
            feats = np.array(item["features"], dtype=np.float32)
            if len(feats) != FEATURE_COUNT:
                continue
            sample = {
                "features":    feats,
                "true_label":  int(item.get("true_label", 0)),
                "attack_type": item.get("attack_type", "REAL"),
                "source_ip":   item.get("source_ip", "?"),
                "dest_port":   int(item.get("dest_port", 0)),
                "timestamp":   item.get("timestamp", datetime.now().strftime("%H:%M:%S")),
            }
            row = process_sample(sample, source="real")
            st.session_state.log.insert(0, row)
            st.session_state.real_device_count += 1
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# TICK — called every rerun while running
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.running:
    mode = st.session_state.traffic_mode

    if mode in ("Simulated", "Mixed"):
        sim_mode = "attack" if random.random() < (attack_mix / 100) else "benign"
        sample   = generate_traffic_sample(sim_mode)
        row      = process_sample(sample, source="sim")
        st.session_state.log.insert(0, row)

    if mode in ("Real Device", "Mixed"):
        ingest_real_traffic()

    if len(st.session_state.log) > 500:
        st.session_state.log = st.session_state.log[:500]

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_dash, tab_real, tab_attack, tab_help = st.tabs(
    ["📊 Dashboard", "🌐 Real Traffic", "💥 Attack Device", "📖 Help"]
)

# ═════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═════════════════════════════════════════════
with tab_dash:
    st.title("🛡️ Real-time Intrusion Detection")
    st.markdown(
        f"Model: **{model_choice}** &nbsp;·&nbsp; "
        f"Features: **{model.num_features}** &nbsp;·&nbsp; "
        f"Threshold: **{st.session_state.threshold:.2f}** &nbsp;·&nbsp; "
        f"Mode: **{st.session_state.traffic_mode}** &nbsp;·&nbsp; "
        f"Real pkts: **{st.session_state.real_device_count}**"
    )

    # Status
    if st.session_state.running:
        st.markdown('<span class="live-dot"></span>**Live** — classifying traffic', unsafe_allow_html=True)
    elif st.session_state.log:
        st.warning("⏸ Paused — press ▶ Start to resume")
    else:
        st.info("Press ▶ Start in the sidebar to begin.")

    # Alert banner
    alert = st.session_state.last_alert
    if alert:
        src_badge = (
            '<span class="device-badge badge-real">REAL</span>'
            if alert.get("src") == "real" else
            '<span class="device-badge badge-inject">INJECT</span>'
            if alert.get("src") == "inject" else
            '<span class="device-badge badge-sim">SIM</span>'
        )
        st.markdown(
            f'<div class="attack-alert">⚠️ LAST ATTACK &nbsp;·&nbsp; '
            f'{src_badge} &nbsp;·&nbsp; '
            f'IP: <code>{alert["source_ip"]}</code> &nbsp;·&nbsp; '
            f'Type: <strong>{alert["attack_type"]}</strong> &nbsp;·&nbsp; '
            f'Conf: <strong>{alert["confidence"]*100:.1f}%</strong>'
            f' &nbsp;·&nbsp; {alert["time"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="benign-ok">✓ No attacks detected — all traffic normal</div>',
            unsafe_allow_html=True,
        )

    # Live net I/O (psutil)
    if _PSUTIL and st.session_state.running:
        net  = _psutil.net_io_counters()
        prev = st.session_state.prev_net
        d_s = net.bytes_sent - prev["bytes_sent"] if prev else 0
        d_r = net.bytes_recv - prev["bytes_recv"] if prev else 0
        st.session_state.prev_net = {
            "bytes_sent": net.bytes_sent, "bytes_recv": net.bytes_recv,
            "pkts": net.packets_sent + net.packets_recv,
        }
        st.markdown(
            f'<div class="net-stat-box">🌐 <b>Host net I/O</b> &nbsp;|&nbsp; '
            f'↑ {net.bytes_sent/1e6:.1f} MB  ↓ {net.bytes_recv/1e6:.1f} MB '
            f'&nbsp;|&nbsp; Δtick: ↑ {d_s/1024:.1f} KB ↓ {d_r/1024:.1f} KB'
            f'</div>', unsafe_allow_html=True,
        )

    st.markdown("---")

    # Confusion matrix cards
    tp = st.session_state.tp; tn = st.session_state.tn
    fp = st.session_state.fp; fn = st.session_state.fn
    total = tp + tn + fp + fn

    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall    = tp/(tp+fn) if (tp+fn) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    acc       = (tp+tn)/total if total else 0.0

    st.markdown("### 📊 Session Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="cm-tp"><div class="cm-val" style="color:#16A34A">{tp}</div><div class="cm-lbl" style="color:#166534">✅ True Positives</div><div class="cm-sub">Attack → correctly flagged</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="cm-tn"><div class="cm-val" style="color:#2563EB">{tn}</div><div class="cm-lbl" style="color:#1E40AF">✅ True Negatives</div><div class="cm-sub">Benign → correctly cleared</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="cm-fp"><div class="cm-val" style="color:#D97706">{fp}</div><div class="cm-lbl" style="color:#92400E">⚠️ False Positives</div><div class="cm-sub">Benign → wrongly flagged</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="cm-fn"><div class="cm-val" style="color:#DC2626">{fn}</div><div class="cm-lbl" style="color:#7F1D1D">❌ False Negatives</div><div class="cm-sub">Attack → missed!</div></div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc*100:.1f}%")
    m2.metric("Precision", f"{precision*100:.1f}%")
    m3.metric("Recall",    f"{recall*100:.1f}%")
    m4.metric("F1",        f"{f1:.3f}")
    m5.metric("Total",     f"{total}")

    if total > 0:
        with st.expander("📋 2×2 confusion matrix"):
            cm_df = pd.DataFrame(
                {"Predicted BENIGN": [tn, fn], "Predicted ATTACK": [fp, tp]},
                index=["Actual BENIGN", "Actual ATTACK"],
            )
            st.dataframe(
                cm_df.style.background_gradient(cmap="RdYlGn", axis=None).format("{:,}"),
                width="stretch",
            )

    st.markdown("---")

    # Traffic log
    st.markdown("### 📋 Traffic Log (last 100)")
    if not st.session_state.log:
        st.info("Waiting for traffic… Press ▶ Start in the sidebar.")
    else:
        df = pd.DataFrame(st.session_state.log[:100])

        def _badge(origin):
            if origin == "real":
                return "🌐 real"
            elif origin == "inject":
                return "💥 inject"
            return "🔵 sim"

        df["Origin"] = df["Origin"].apply(_badge)

        def _style(row):
            s   = [""] * len(row)
            idx = list(row.index)
            p   = idx.index("Predicted")
            c   = idx.index("✓/✗")
            tl  = idx.index("True label")
            s[p] = ("background:#FEE2E2;color:#7F1D1D;font-weight:bold"
                    if row["Predicted"] == "ATTACK"
                    else "background:#F0FDF4;color:#14532D")
            if row["✓/✗"] == "✗":
                s[c] = "background:#FEF9C3;color:#713F12;font-weight:bold"
            s[tl] = ("background:#FEF2F2" if row["True label"] == "ATTACK"
                     else "background:#F0F9FF")
            return s

        show_cols = ["Time", "Source IP", "Port", "Origin",
                     "True label", "Predicted", "Confidence", "P(Atk)", "✓/✗"]
        st.dataframe(
            df[show_cols].style.apply(_style, axis=1),
            width="stretch", height=340,
        )

    if len(st.session_state.log) >= 5:
        with st.expander("📈 Detection over time"):
            df_c = pd.DataFrame(st.session_state.log).head(100)
            st.line_chart(pd.DataFrame({
                "Attacks caught":    (df_c["Predicted"] == "ATTACK").cumsum().values,
                "Benign classified": (df_c["Predicted"] == "BENIGN").cumsum().values,
            }), height=180)

        with st.expander("🔬 Probability distribution"):
            raw_p   = [float(r.replace("%",""))/100 for r in
                       pd.DataFrame(st.session_state.log).head(100)["P(Atk)"]]
            thr_val = st.session_state.threshold
            above   = sum(1 for p in raw_p if p >= thr_val)
            st.caption(f"Threshold = **{thr_val:.2f}** · {above}/{len(raw_p)} above ({above/max(len(raw_p),1)*100:.0f}% ATTACK)")
            st.bar_chart(pd.DataFrame({"P(Attack)": raw_p}), height=150)


# ═════════════════════════════════════════════
# TAB 2 — REAL TRAFFIC SETUP
# ═════════════════════════════════════════════
with tab_real:
    st.header("🌐 Real Network Traffic")

    st.markdown("""
    This tab explains how to classify **actual packets** from your machine, phone, or 
    another device on the same network. The model was trained on CIC-IDS2017 flow-level 
    statistics — so you need to capture packets and extract 76 numerical features per flow.

    There are two approaches depending on where you're running the dashboard.
    """)

    col_local, col_colab = st.columns(2)

    with col_local:
        st.subheader("🖥️ Running Locally (recommended)")
        st.markdown("""
**Best for real traffic** — you can capture packets directly.

**Step 1 — Install nfstream**
```bash
pip install nfstream pandas requests
```

**Step 2 — Run `capture_agent.py`** (provided below)
```bash
# On the same machine as the dashboard:
python live_demo/capture_agent.py --interface eth0

# From a second laptop / phone on the same WiFi
# (point it at your dashboard machine's IP):
python capture_agent.py --interface wlan0 \\
       --server http://192.168.1.X:8502
```

**Step 3 — Set Traffic Source to "Real Device"** in the sidebar and press ▶ Start.

The agent captures live packets, extracts features matching your trained 
`feature_names.txt`, scales them with your `scaler.pkl`, and POSTs them to 
the dashboard queue. The dashboard picks them up on every tick.
        """)

    with col_colab:
        st.subheader("☁️ Running on Colab")
        st.markdown("""
**Colab can't capture packets** — it runs in Google's cloud, not on your local network.

**What you CAN do on Colab:**

✅ Use `capture_agent.py` on your **local machine** to send traffic  
✅ Use `--server` flag pointing at your ngrok URL  
✅ Upload a pre-captured `.pcap` and extract features  
✅ Upload a CSV with flow features directly  

**Upload CSV flow → classify:**
        """)
        pcap_up = st.file_uploader("Upload a pre-extracted flow CSV (76 features)", type=["csv"], key="real_csv")
        if pcap_up:
            import joblib
            data_dir = _ROOT / "data" / "processed"
            try:
                scaler = joblib.load(data_dir / "scaler.pkl")
                with open(data_dir / "feature_names.txt") as f:
                    feats = [l.strip() for l in f if l.strip()]
                df_up = pd.read_csv(pcap_up, low_memory=False)
                df_up.columns = df_up.columns.str.strip()
                miss = [c for c in feats if c not in df_up.columns]
                if miss:
                    st.error(f"Missing {len(miss)} columns: {miss[:5]}…")
                else:
                    X_up = df_up[feats].select_dtypes(include=[np.number]).values.astype(np.float32)
                    X_sc = scaler.transform(X_up)
                    results = model.classify_batch(X_sc)
                    df_up["Prediction"]  = [r["prediction"] for r in results]
                    df_up["Confidence"]  = [f"{r['confidence']*100:.1f}%" for r in results]
                    df_up["P(Attack)"]   = [f"{r['prob_attack']*100:.1f}%" for r in results]
                    preds = [r["label"] for r in results]
                    st.success(f"Classified {len(preds):,} flows — "
                               f"{sum(preds)} ATTACK ({sum(preds)/max(len(preds),1)*100:.1f}%)")
                    st.dataframe(df_up[["Prediction","Confidence","P(Attack)"]].head(200),
                                 width="stretch")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("📡 Queue status")
    q_size = 0
    if QUEUE_FILE.exists():
        try:
            with open(QUEUE_FILE) as f:
                q_size = len(json.load(f))
        except Exception:
            pass
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Queue file", str(QUEUE_FILE.name))
    col_b.metric("Queued packets", q_size)
    col_c.metric("Real packets classified", st.session_state.real_device_count)
    if st.button("🗑 Flush queue"):
        if QUEUE_FILE.exists():
            QUEUE_FILE.write_text("[]")
        st.success("Queue flushed.")

    # Colab vs local comparison
    st.markdown("---")
    st.subheader("Colab vs Local — which should you use?")
    cmp = {
        "Feature": ["Real packet capture", "Attack device simulation",
                    "Dashboard access", "GPU training", "Session persistence",
                    "Setup effort", "Recommended for"],
        "Local": ["✅ Full (tcpdump/nfstream)", "✅ Any device on LAN",
                  "✅ http://localhost:8501", "❌ CPU only (slow)", "✅ Files stay on disk",
                  "Low", "Live demos, real traffic"],
        "Colab": ["❌ Cloud — no LAN access", "✅ Via capture_agent + ngrok URL",
                  "⚠️ Needs ngrok tunnel", "✅ Free T4 GPU", "⚠️ Resets each session",
                  "Medium (ngrok + Drive)", "Training, offline analysis"],
    }
    st.dataframe(pd.DataFrame(cmp).set_index("Feature"), width="stretch")


# ═════════════════════════════════════════════
# TAB 3 — ATTACK DEVICE
# ═════════════════════════════════════════════
with tab_attack:
    st.header("💥 Simulated Attack Device")
    st.markdown("""
    Simulate a remote device launching specific attacks at your IDS.
    Each injection crafts a realistic feature vector matching the attack's known
    statistical signature from CIC-IDS2017, then feeds it directly to the model.
    """)

    st.markdown("### Single injection")
    col1, col2 = st.columns([2, 1])
    with col1:
        attack_type = st.selectbox(
            "Attack type",
            list(ATTACK_PROFILES.keys()),
            format_func=lambda k: f"{k} — {ATTACK_PROFILES[k]['description']}",
            key="atk_type_inject",
        )
        n_inject = st.slider("Number of packets to inject", 1, 20, 1, key="n_inject")

    with col2:
        profile = ATTACK_PROFILES[attack_type]
        st.markdown(f"**{attack_type}**")
        st.markdown(f"_{profile['description']}_")
        st.markdown(f"Simulated source: `{profile['source_ip'].format(r=42)}`")

        if st.button("⚡ Inject now", type="primary", key="inject_primary"):
            for _ in range(n_inject):
                feats = _attack_vector(attack_type)
                r     = random.randint(1, 254)
                sample = {
                    "features":    feats,
                    "true_label":  1,
                    "attack_type": attack_type,
                    "source_ip":   profile["source_ip"].format(r=r),
                    "dest_port":   int(feats[0]),
                    "timestamp":   datetime.now().strftime("%H:%M:%S.%f")[:-3],
                }
                row = process_sample(sample, source="inject")
                st.session_state.log.insert(0, row)
            if len(st.session_state.log) > 500:
                st.session_state.log = st.session_state.log[:500]
            st.success(f"Injected {n_inject}× **{attack_type}** → check Dashboard tab")
            st.rerun()

    st.markdown("---")
    st.markdown("### Burst scenario")
    st.markdown("Simulate a multi-stage attack campaign from a single 'attacker device'.")

    scenario = st.selectbox(
        "Scenario",
        ["Recon → Escalate → Flood", "BruteForce → Botnet", "PortScan → WebAttack"],
        key="scenario_sel",
    )
    burst_n = st.slider("Packets per stage", 2, 10, 3, key="burst_n")

    SCENARIOS = {
        "Recon → Escalate → Flood":  ["PortScan", "BruteForce", "DDoS"],
        "BruteForce → Botnet":       ["BruteForce", "Botnet"],
        "PortScan → WebAttack":      ["PortScan", "WebAttack"],
    }

    if st.button("🚨 Launch scenario", type="primary", key="burst_launch"):
        stages = SCENARIOS[scenario]
        total_injected = 0
        r = random.randint(1, 254)
        for stage_name in stages:
            prof = ATTACK_PROFILES[stage_name]
            for _ in range(burst_n):
                feats  = _attack_vector(stage_name)
                sample = {
                    "features":    feats,
                    "true_label":  1,
                    "attack_type": f"{stage_name} [{scenario.split('→')[0].strip()} campaign]",
                    "source_ip":   prof["source_ip"].format(r=r),
                    "dest_port":   int(feats[0]),
                    "timestamp":   datetime.now().strftime("%H:%M:%S.%f")[:-3],
                }
                row = process_sample(sample, source="inject")
                st.session_state.log.insert(0, row)
                total_injected += 1
        if len(st.session_state.log) > 500:
            st.session_state.log = st.session_state.log[:500]
        st.success(f"Scenario **{scenario}** — {total_injected} packets injected across "
                   f"{len(stages)} stages → check Dashboard tab")
        st.rerun()

    st.markdown("---")
    st.subheader("How to simulate from a REAL external device")
    st.markdown("""
    If you want a genuine second machine (another laptop, a Raspberry Pi, 
    or even an Android phone with Termux) to simulate attacks:

    **Option A — use `capture_agent.py` with `--inject` flag**
    ```bash
    # On the attacker device (must be on the same LAN as the dashboard):
    pip install requests numpy
    python capture_agent.py --mode inject \\
           --attack DDoS --n 50 \\
           --server http://192.168.1.YOUR_DASHBOARD_IP:8502
    ```
    The agent will craft DDoS-profile vectors and POST them to the dashboard queue.

    **Option B — nmap port scan (generates REAL traffic to capture)**
    ```bash
    # On attacker device — harmless port scan of your dashboard machine
    nmap -sS 192.168.1.YOUR_DASHBOARD_IP
    ```
    Then run `capture_agent.py --interface eth0` on the dashboard machine 
    to capture and classify those real packets.

    **Option C — HTTP flood (Python)**
    ```python
    import requests, time
    for i in range(200):
        try: requests.get("http://192.168.1.YOUR_IP", timeout=0.3)
        except: pass
    ```
    Capture with `capture_agent.py` → model should flag as DDoS/BruteForce.
    """)


# ═════════════════════════════════════════════
# TAB 4 — HELP
# ═════════════════════════════════════════════
with tab_help:
    st.header("📖 Help & Architecture")

    st.markdown("""
    ### Data flow
    ```
    External device           Dashboard machine            Model
    ──────────────            ─────────────────            ─────
    capture_agent.py  ──►  .traffic_queue.json  ──►  IDS_Model.classify()
    (packets → feats)       (JSON queue file)          (PyTorch inference)
         or
    traffic_simulator.py ──► direct call ──────────►  IDS_Model.classify()
    (synthetic vectors)
    ```

    ### Traffic modes
    | Mode | What happens |
    |---|---|
    | **Simulated** | Vectors built by `traffic_simulator.py` — no real packets needed |
    | **Real Device** | Reads from queue file populated by `capture_agent.py` |
    | **Mixed** | One simulated + all queued real packets per tick |

    ### Threshold guide
    | Threshold | Effect |
    |---|---|
    | 0.3–0.4 | Aggressive — catches more attacks, more false alarms |
    | 0.5 | Balanced default |
    | 0.6–0.7 | Conservative — fewer false alarms, may miss some attacks |
    | **Auto-tune** | Picks the value that maximises F1 on your test set |

    ### Quick start commands
    ```bash
    # Local — full pipeline
    pip install -r requirements.txt
    python src/data/preprocess_cicids2017.py
    python run_all.py
    streamlit run live_demo/live_demo.py

    # Capture real traffic (separate terminal)
    python live_demo/capture_agent.py --interface eth0

    # Colab — training only
    # Open notebooks/IDS_Colab_Notebook.ipynb → Runtime → T4 GPU → Run all
    ```

    ### Files
    | File | Purpose |
    |---|---|
    | `live_demo/live_demo.py` | This dashboard |
    | `live_demo/capture_agent.py` | Packet capture + feature extraction agent |
    | `live_demo/traffic_simulator.py` | Synthetic traffic generator |
    | `live_demo/model_loader.py` | Loads .pt checkpoint + scaler |
    | `app.py` | Separate static results dashboard |
    | `run_all.py` | Train all 4 models |
    """)

# ── Rerun trigger ──────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(delay)
    st.rerun()
