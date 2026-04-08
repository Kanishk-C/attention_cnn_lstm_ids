"""
capture_agent.py  —  IDS Capture & Inject Agent
================================================
Run this on ANY device (phone, laptop, Raspberry Pi, VM) to send
traffic to the live_demo dashboard.

Two modes
─────────
1. CAPTURE mode  (default)
   Captures live packets on a network interface, extracts 76-feature
   flow statistics using nfstream, scales them with the project scaler,
   and POSTs them to the dashboard queue.

   Requires:  pip install nfstream pandas requests joblib numpy

2. INJECT mode
   Crafts synthetic attack vectors matching a named profile and POSTs
   them directly to the dashboard queue — no packet capture needed.
   Useful when running from a device that can't run nfstream
   (e.g. Android/Termux, Windows without Npcap).

   Requires:  pip install requests numpy

Usage
─────
  # Capture real traffic on this machine (writes to shared queue file):
  python capture_agent.py --interface eth0

  # Capture and POST to a remote dashboard (Colab + ngrok):
  python capture_agent.py --interface wlan0 \\
         --server https://abc123.ngrok.io

  # Inject DDoS vectors into a remote dashboard:
  python capture_agent.py --mode inject --attack DDoS --n 30 \\
         --server http://192.168.1.42:8502

  # List available interfaces:
  python capture_agent.py --list-interfaces

  # Dry-run (print features, don't send):
  python capture_agent.py --interface eth0 --dry-run
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

# Suppress sklearn feature names warning
warnings.filterwarnings("ignore", category=UserWarning)

# ── Resolve project root ─────────────────────────────────────────────────────
# _ROOT is the project root (contains data/, app/, training/).
# The old check looked for src/ at the root which no longer exists there.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
# Validate: data/processed must be here — if not, something is very wrong
if not (_ROOT / "data" / "processed").exists():
    raise RuntimeError(
        f"Cannot locate project root. Expected data/processed at {_ROOT}."
        f" Run from the project root: python app/capture_agent.py"
    )

QUEUE_FILE   = _HERE / ".traffic_queue.json"
FEAT_FILE    = _ROOT / "data" / "processed" / "feature_names.txt"
SCALER_FILE  = _ROOT / "data" / "processed" / "scaler.pkl"

# CIC-IDS2017 attack-profile feature vectors (same as traffic_simulator.py)
ATTACK_PROFILES = {
    "DDoS": {
        "description": "Distributed Denial of Service",
        "source_ip": "192.168.100.{r}",
        "features": {
            0: (50, 5), 1: (500, 50), 2: (5000, 200), 3: (0, 0),
            4: (250000, 10000), 5: (0, 0), 6: (50, 2), 7: (0, 0),
            12: (10000, 500), 13: (500000, 20000), 14: (0.05, 0.01),
            22: (1, 0), 23: (0, 0), 25: (0, 0), 26: (5000, 200),
            27: (5000, 200), 60: (0, 0), 61: (0, 0),
        },
    },
    "PortScan": {
        "description": "Network Port Scanning",
        "source_ip": "10.0.{r}.1",
        "features": {
            0: (0, 0), 1: (1000, 100), 2: (1, 0), 3: (0, 0),
            4: (60, 5), 5: (0, 0), 12: (1, 0), 13: (60, 5),
            14: (0, 0), 23: (1, 0), 25: (0, 0), 60: (65535, 0),
        },
    },
    "BruteForce": {
        "description": "SSH/FTP Brute Force",
        "source_ip": "172.16.{r}.50",
        "features": {
            0: (22, 0), 1: (50000, 5000), 2: (30, 5), 3: (30, 5),
            4: (3000, 300), 5: (6000, 600), 6: (100, 10), 7: (200, 20),
            12: (1, 0.1), 13: (150, 15), 22: (5, 1), 27: (30, 5),
        },
    },
    "WebAttack": {
        "description": "SQL Injection / XSS",
        "source_ip": "203.0.{r}.100",
        "features": {
            0: (80, 0), 1: (100000, 10000), 2: (10, 2), 3: (10, 2),
            4: (5000, 500), 5: (2000, 200), 6: (500, 50), 7: (200, 20),
            12: (0.1, 0.01), 13: (70, 7),
        },
    },
    "Botnet": {
        "description": "Botnet C&C Communication",
        "source_ip": "192.168.50.{r}",
        "features": {
            0: (6667, 0), 1: (300000, 30000), 2: (50, 10), 3: (50, 10),
            4: (5000, 500), 5: (5000, 500), 6: (100, 10), 7: (100, 10),
            14: (6000, 600), 27: (50, 10),
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_scaler():
    if not SCALER_FILE.exists():
        return None
    try:
        import joblib
        return joblib.load(SCALER_FILE)
    except Exception as e:
        print(f"[WARN] Could not load scaler: {e}")
        return None


def _load_feature_names():
    if not FEAT_FILE.exists():
        return []
    with open(FEAT_FILE, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def _make_attack_vector(profile_name: str, n_features: int) -> np.ndarray:
    profile = ATTACK_PROFILES[profile_name]
    v = np.zeros(n_features, dtype=np.float32)
    for idx, (mean, std) in profile["features"].items():
        if idx < n_features:
            v[idx] = max(0.0, random.gauss(mean, std) if std > 0 else float(mean))
    for i in range(n_features):
        if v[i] == 0 and i not in profile["features"]:
            v[i] = max(0.0, random.gauss(0.5, 0.5))
    return v


def _push_local(records: list):
    """Write records to the shared queue JSON file (same machine as dashboard)."""
    existing = []
    if QUEUE_FILE.exists():
        try:
            with open(QUEUE_FILE) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.extend(records)
    with open(QUEUE_FILE, "w") as f:
        json.dump(existing, f)
    try:
        os.chmod(QUEUE_FILE, 0o666)  # Give rw permissions so streamlit user can clear it
    except Exception:
        pass


def _push_remote(records: list, server_url: str):
    """POST records to a remote dashboard endpoint."""
    import requests
    url = server_url.rstrip("/") + "/traffic_queue"
    try:
        resp = requests.post(url, json=records, timeout=5)
        if resp.status_code != 200:
            print(f"[WARN] Server returned {resp.status_code}")
    except requests.exceptions.ConnectionError:
        # Fall back to local queue file if unreachable
        print("[WARN] Remote server unreachable — writing to local queue file")
        _push_local(records)


def _send(records: list, server_url: str | None, dry_run: bool):
    if dry_run:
        for r in records:
            feats = r["features"][:6]
            print(f"[DRY-RUN] {r['timestamp']}  {r['source_ip']:<20} "
                  f"{r['attack_type']:<12} features[:6]={np.round(feats, 2)}")
        return
    if server_url:
        _push_remote(records, server_url)
    else:
        _push_local(records)


# ══════════════════════════════════════════════════════════════════════════════
# INJECT MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_inject(attack: str, n: int, server_url, dry_run: bool):
    scaler = _load_scaler()
    n_feats = (int(scaler.n_features_in_)
               if scaler and hasattr(scaler, "n_features_in_") else 76)

    if attack not in ATTACK_PROFILES:
        print(f"[ERROR] Unknown attack '{attack}'. Choose from: {list(ATTACK_PROFILES)}")
        sys.exit(1)

    profile = ATTACK_PROFILES[attack]
    print(f"[INJECT] {n}× {attack} — {profile['description']}")

    records = []
    for _ in range(n):
        feats = _make_attack_vector(attack, n_feats)
        # Ensure feats is float32 without scaling
        feats = feats.astype(np.float32)
        r = random.randint(1, 254)
        records.append({
            "features":    feats.tolist(),
            "true_label":  1,
            "attack_type": attack,
            "source_ip":   profile["source_ip"].format(r=r),
            "dest_port":   int(feats[0]),
            "timestamp":   datetime.now().strftime("%H:%M:%S.%f")[:-3],
        })

    _send(records, server_url, dry_run)
    print(f"[INJECT] Sent {len(records)} records.")


# ══════════════════════════════════════════════════════════════════════════════
# CAPTURE MODE
# ══════════════════════════════════════════════════════════════════════════════

def _nfstream_available():
    try:
        import nfstream  # noqa
        return True
    except ImportError:
        return False


def _map_nfstream_to_cicids(flow, feat_names: list, n_feats: int) -> np.ndarray | None:
    """
    Map an nfstream flow object to a CIC-IDS2017-compatible feature vector.

    nfstream and CIC-IDS2017 use different column names.  This function covers
    the most common mappings.  Unknown features default to 0.
    """
    # nfstream attribute → CIC-IDS2017 feature name mappings
    MAPPING = {
        "dst_port":                "Destination Port",
        "bidirectional_duration_ms": "Flow Duration",
        "src2dst_packets":         "Total Fwd Packets",
        "dst2src_packets":         "Total Backward Packets",
        "src2dst_bytes":           "Total Length of Fwd Packets",
        "dst2src_bytes":           "Total Length of Bwd Packets",
        "src2dst_mean_piat_ms":    "Fwd IAT Mean",
        "dst2src_mean_piat_ms":    "Bwd IAT Mean",
        "bidirectional_mean_piat_ms": "Flow IAT Mean",
        "src2dst_syn_packets":     "SYN Flag Count",
        "src2dst_ack_packets":     "ACK Flag Count",
        "src2dst_fin_packets":     "FIN Flag Count",
        "src2dst_rst_packets":     "RST Flag Count",
        "src2dst_psh_packets":     "PSH Flag Count",
        "src2dst_urg_packets":     "URG Flag Count",
    }
    v = np.zeros(n_feats, dtype=np.float32)
    for nf_attr, cic_name in MAPPING.items():
        val = getattr(flow, nf_attr, None)
        if val is None:
            continue
        try:
            idx = feat_names.index(cic_name)
            v[idx] = max(0.0, float(val))
        except ValueError:
            pass
    return v


def run_capture(interface: str, server_url, dry_run: bool,
                interval_s: float, max_flows: int):
    if not _nfstream_available():
        print("[ERROR] nfstream not installed.")
        print("        pip install nfstream pandas")
        sys.exit(1)

    from nfstream import NFStreamer

    scaler     = _load_scaler()
    feat_names = _load_feature_names()
    n_feats    = (int(scaler.n_features_in_)
                  if scaler and hasattr(scaler, "n_features_in_") else 76)

    print(f"[CAPTURE] Interface: {interface}")
    print(f"[CAPTURE] Feature count: {n_feats}")
    print(f"[CAPTURE] Scaler: {'loaded' if scaler else 'NOT FOUND (features unscaled!)'}")
    print(f"[CAPTURE] Destination: {'DRY-RUN' if dry_run else (server_url or 'local queue file')}")
    print(f"[CAPTURE] Press Ctrl+C to stop.\n")

    total_sent = 0
    while True:
        try:
            streamer = NFStreamer(
                source=interface,
                statistical_analysis=True,
                idle_timeout=int(interval_s),
                active_timeout=int(interval_s * 2),
                max_nflows=max_flows,
            )
            records = []
            for flow in streamer:
                feats = _map_nfstream_to_cicids(flow, feat_names, n_feats)
                if feats is None:
                    continue
                # Ensure feats is float32 without scaling
                feats = feats.astype(np.float32)
                records.append({
                    "features":    feats.tolist(),
                    "true_label":  0,          # real traffic — assume benign
                    "attack_type": "REAL",
                    "source_ip":   str(getattr(flow, "src_ip", "?")),
                    "dest_port":   int(getattr(flow, "dst_port", 0)),
                    "timestamp":   datetime.now().strftime("%H:%M:%S.%f")[:-3],
                })

            if records:
                _send(records, server_url, dry_run)
                total_sent += len(records)
                print(f"[CAPTURE] Sent {len(records)} flows (total: {total_sent})")
            else:
                print("[CAPTURE] No flows captured this interval.")

            time.sleep(interval_s)

        except KeyboardInterrupt:
            print(f"\n[CAPTURE] Stopped. Total flows sent: {total_sent}")
            break
        except Exception as e:
            print(f"[CAPTURE] Error: {e}")
            time.sleep(interval_s)


# ══════════════════════════════════════════════════════════════════════════════
# LIST INTERFACES
# ══════════════════════════════════════════════════════════════════════════════

def list_interfaces():
    print("Available network interfaces:\n")
    # Try psutil first
    try:
        import psutil
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        for name, snic_list in addrs.items():
            st    = stats.get(name)
            up    = "UP" if (st and st.isup) else "DOWN"
            speed = f"{st.speed} Mb/s" if (st and st.speed > 0) else ""
            ips   = [s.address for s in snic_list if "." in s.address or ":" in s.address]
            print(f"  {name:<20} {up:<6} {speed:<12} {', '.join(ips)}")
        return
    except ImportError:
        pass

    # Fallback — read /proc/net/dev
    try:
        with open("/proc/net/dev") as f:
            lines = f.readlines()[2:]
        for line in lines:
            iface = line.split(":")[0].strip()
            print(f"  {iface}")
    except Exception:
        print("  Could not list interfaces. Install psutil: pip install psutil")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IDS Capture & Inject Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["capture", "inject"], default="capture",
                        help="capture: live packets | inject: synthetic attack vectors")
    parser.add_argument("--interface", "-i", default="eth0",
                        help="Network interface to capture (capture mode)")
    parser.add_argument("--attack", "-a", default="DDoS",
                        choices=list(ATTACK_PROFILES.keys()),
                        help="Attack profile to inject (inject mode)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of packets to inject (inject mode)")
    parser.add_argument("--server", "-s", default=None,
                        help="Dashboard URL, e.g. http://192.168.1.42:8501 "
                             "(omit to write directly to local queue file)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Capture interval in seconds (capture mode)")
    parser.add_argument("--max-flows", type=int, default=50,
                        help="Max flows per capture interval")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print features but do not send to dashboard")
    parser.add_argument("--list-interfaces", action="store_true",
                        help="List network interfaces and exit")
    args = parser.parse_args()

    if args.list_interfaces:
        list_interfaces()
        return

    if args.mode == "inject":
        run_inject(
            attack     = args.attack,
            n          = args.n,
            server_url = args.server,
            dry_run    = args.dry_run,
        )
    else:
        run_capture(
            interface  = args.interface,
            server_url = args.server,
            dry_run    = args.dry_run,
            interval_s = args.interval,
            max_flows  = args.max_flows,
        )


if __name__ == "__main__":
    main()
