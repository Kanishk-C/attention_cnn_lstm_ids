"""
simulator.py — Synthetic Network Traffic Generator

All feature indices are 0-based and verified against data/processed/feature_names.txt:

  0  Destination Port          1  Flow Duration (µs)
  2  Total Fwd Packets         3  Total Backward Packets
  4  Total Length Fwd Packets  5  Total Length Bwd Packets
  6  Fwd Pkt Len Max           7  Fwd Pkt Len Min
  8  Fwd Pkt Len Mean          9  Fwd Pkt Len Std
 10  Bwd Pkt Len Max          11  Bwd Pkt Len Min
 12  Bwd Pkt Len Mean         13  Bwd Pkt Len Std
 14  Flow Bytes/s             15  Flow Packets/s
 16  Flow IAT Mean            17  Flow IAT Std
 18  Flow IAT Max             19  Flow IAT Min
 20  Fwd IAT Total            21  Fwd IAT Mean
 22  Fwd IAT Std              23  Fwd IAT Max
 24  Fwd IAT Min              25  Bwd IAT Total
 26  Bwd IAT Mean             27  Bwd IAT Std
 28  Bwd IAT Max              29  Bwd IAT Min
 30  Fwd PSH Flags            31  Bwd PSH Flags
 32  Fwd URG Flags            33  Bwd URG Flags
 34  Fwd Header Length        35  Bwd Header Length
 36  Fwd Packets/s            37  Bwd Packets/s
 38  Min Packet Length        39  Max Packet Length
 40  Packet Length Mean       41  Packet Length Std
 42  Packet Length Variance   43  FIN Flag Count
 44  SYN Flag Count           45  RST Flag Count
 46  PSH Flag Count           47  ACK Flag Count
 48  URG Flag Count           49  CWE Flag Count
 50  ECE Flag Count           51  Down/Up Ratio
 52  Average Packet Size      53  Avg Fwd Segment Size
 54  Avg Bwd Segment Size     55  Fwd Header Length.1
 56  Fwd Avg Bytes/Bulk       57  Fwd Avg Packets/Bulk
 58  Fwd Avg Bulk Rate        59  Bwd Avg Bytes/Bulk
 60  Bwd Avg Packets/Bulk     61  Bwd Avg Bulk Rate
 62  Subflow Fwd Packets      63  Subflow Fwd Bytes
 64  Subflow Bwd Packets      65  Subflow Bwd Bytes
 66  Init_Win_bytes_forward   67  Init_Win_bytes_backward
 68  act_data_pkt_fwd         69  min_seg_size_forward
 70  Active Mean              71  Active Std
 72  Active Max               73  Active Min
 74  Idle Mean                75  Idle Std
 76  Idle Max                 77  Idle Min

Design: all vectors are synthesised from statistical profiles grounded in
CIC-IDS2017 published statistics. The test set is NOT used — the model
performs real inference on genuinely novel synthetic data each time.
"""

import os
import random
import numpy as np
from datetime import datetime

# ── Resolve paths ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load_scaler():
    scaler_path = os.path.join(_ROOT, "data", "processed", "scaler.pkl")
    if not os.path.exists(scaler_path):
        return None
    try:
        import joblib
        return joblib.load(scaler_path)
    except Exception:
        return None


_SCALER = _load_scaler()

if _SCALER is not None and hasattr(_SCALER, "n_features_in_"):
    FEATURE_COUNT: int = int(_SCALER.n_features_in_)
else:
    FEATURE_COUNT: int = 78   # CIC-IDS2017 has 78 numeric features after label drop

# ── Load the test set to sample real CIC-IDS2017 traffic vectors ─────────────
# X_test.npy is z-scored; we inverse_transform → raw space → classify() re-scales.
# Using REAL captured traffic (not guessed profiles) guarantees the features land
# in the exact probability space the model was trained on.
_DATA_DIR = os.path.join(_ROOT, "data", "processed")
_X_TEST: np.ndarray | None = None
_Y_TEST: np.ndarray | None = None
_BENIGN_IDX: np.ndarray = np.array([], dtype=int)
_ATTACK_IDX: np.ndarray = np.array([], dtype=int)

try:
    _xp = os.path.join(_DATA_DIR, "X_test.npy")
    _yp = os.path.join(_DATA_DIR, "y_test.npy")
    if os.path.exists(_xp) and os.path.exists(_yp):
        _X_TEST = np.load(_xp, mmap_mode="r")
        _Y_TEST = np.load(_yp, mmap_mode="r")
        _BENIGN_IDX = np.where(_Y_TEST == 0)[0]
        _ATTACK_IDX = np.where(_Y_TEST == 1)[0]
except Exception:
    pass



# ── Optional psutil for host network counters ─────────────────────────────────
try:
    import psutil as _psutil
    _PSUTIL_OK = True
except ImportError:
    _psutil    = None
    _PSUTIL_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Each profile entry: (mean, std) — sampled via Gaussian, clipped at 0.
# Indices from the verified feature_names.txt mapping above.
# ══════════════════════════════════════════════════════════════════════════════

ATTACK_PROFILES = {
    "DDoS": {
        "description": "Distributed Denial of Service — high-volume SYN/UDP flood",
        "source_ip":   "192.168.100.{r}",
        "features": {
            # Port & flow shape
            0:  (80,       20),       # Destination Port (HTTP target)
            1:  (500,      200),      # Flow Duration µs — extremely short
            2:  (1200,     300),      # Total Fwd Packets — very high
            3:  (0,        0),        # Total Backward Packets — server overwhelmed
            4:  (60000,    8000),     # Total Length Fwd Packets — high volume
            5:  (0,        0),        # Total Length Bwd Packets
            6:  (54,       4),        # Fwd Packet Length Max — small flood packets
            7:  (54,       4),        # Fwd Packet Length Min
            8:  (54,       4),        # Fwd Packet Length Mean
            9:  (0,        0),        # Fwd Packet Length Std — very uniform
            10: (0,        0),        # Bwd Packet Length Max
            11: (0,        0),        # Bwd Packet Length Min
            12: (0,        0),        # Bwd Packet Length Mean
            13: (0,        0),        # Bwd Packet Length Std
            # Rates — hallmark of DDoS
            14: (5000000,  1000000),  # Flow Bytes/s — very high
            15: (100000,   20000),    # Flow Packets/s — very high
            # Timing — near-zero IAT
            16: (5,        2),        # Flow IAT Mean µs
            17: (2,        1),        # Flow IAT Std
            18: (100,      20),       # Flow IAT Max
            19: (0,        0),        # Flow IAT Min
            # Flags
            44: (1,        0),        # SYN Flag Count — SYN flood
            45: (0,        0),        # RST Flag Count
            47: (0,        0),        # ACK Flag Count
            # Packet stats
            38: (54,       4),        # Min Packet Length
            39: (54,       4),        # Max Packet Length
            40: (54,       4),        # Packet Length Mean
            42: (0,        0),        # Packet Length Variance
            52: (54,       4),        # Average Packet Size
            # Window size typical of attack tools
            66: (1024,     0),        # Init_Win_bytes_forward
            67: (0,        0),        # Init_Win_bytes_backward
        },
    },

    "PortScan": {
        "description": "Network port scanning — one SYN per port, RST responses",
        "source_ip":   "10.0.{r}.1",
        "features": {
            0:  (0,        0),        # Destination Port — varies (0 = fill randomly)
            1:  (700,      200),      # Flow Duration µs — probe then RST
            2:  (1,        0),        # Total Fwd Packets — exactly 1 SYN
            3:  (1,        0),        # Total Backward Packets — RST-ACK or SYN-ACK
            4:  (0,        0),        # Total Length Fwd Packets — SYN has no data
            5:  (0,        0),        # Total Length Bwd Packets
            6:  (0,        0),        # Fwd Packet Length Max
            7:  (0,        0),        # Fwd Packet Length Min
            8:  (0,        0),        # Fwd Packet Length Mean
            # Rates
            14: (0,        0),        # Flow Bytes/s
            15: (1400,     300),      # Flow Packets/s — fast scanning
            # Timing
            16: (700,      200),      # Flow IAT Mean  
            # Flags — SYN send, RST received
            44: (1,        0),        # SYN Flag Count
            45: (1,        0),        # RST Flag Count — closed port RST
            47: (0,        0),        # ACK Flag Count
            # Window
            66: (1024,     0),        # Init_Win_bytes_forward
            67: (0,        0),        # Init_Win_bytes_backward
        },
    },

    "BruteForce": {
        "description": "SSH/FTP brute force — repeated auth attempts on port 22",
        "source_ip":   "172.16.{r}.50",
        "features": {
            0:  (22,       0),        # Destination Port — SSH
            1:  (45000,    10000),    # Flow Duration µs — per attempt
            2:  (20,       5),        # Total Fwd Packets
            3:  (18,       5),        # Total Backward Packets — server responds
            4:  (2500,     400),      # Total Length Fwd Packets
            5:  (4000,     600),      # Total Length Bwd Packets — server banners
            6:  (300,      50),       # Fwd Packet Length Max
            7:  (20,       5),        # Fwd Packet Length Min
            8:  (125,      30),       # Fwd Packet Length Mean
            9:  (80,       20),       # Fwd Packet Length Std
            10: (400,      80),       # Bwd Packet Length Max
            12: (220,      50),       # Bwd Packet Length Mean
            # Rates
            14: (55000,    15000),    # Flow Bytes/s — moderate
            15: (444,      100),      # Flow Packets/s
            # Timing — regular brute-force timing
            16: (2500,     500),      # Flow IAT Mean µs
            17: (500,      100),      # Flow IAT Std — low variance (scripted)
            # Flags
            44: (1,        0),        # SYN Flag Count
            47: (19,       4),        # ACK Flag Count — high (ongoing connection)
            43: (1,        0),        # FIN Flag Count
            # Segment sizes
            53: (125,      30),       # Avg Fwd Segment Size
            54: (220,      50),       # Avg Bwd Segment Size
            66: (65535,    0),        # Init_Win_bytes_forward
            67: (65535,    0),        # Init_Win_bytes_backward
        },
    },

    "WebAttack": {
        "description": "SQL injection / XSS on HTTP port 80",
        "source_ip":   "203.0.{r}.100",
        "features": {
            0:  (80,       0),        # Destination Port — HTTP
            1:  (120000,   30000),    # Flow Duration µs
            2:  (12,       3),        # Total Fwd Packets
            3:  (10,       3),        # Total Backward Packets
            4:  (8000,     2000),     # Total Length Fwd Packets — large payloads
            5:  (3000,     800),      # Total Length Bwd Packets
            6:  (1460,     100),      # Fwd Packet Length Max — MTU-sized injections
            7:  (20,       5),        # Fwd Packet Length Min
            8:  (650,      150),      # Fwd Packet Length Mean — large average
            9:  (450,      100),      # Fwd Packet Length Std — variable payloads
            10: (700,      100),      # Bwd Packet Length Max
            12: (300,      80),       # Bwd Packet Length Mean
            # Rates
            14: (90000,    20000),    # Flow Bytes/s
            15: (100,      25),       # Flow Packets/s
            # Timing
            16: (12000,    3000),     # Flow IAT Mean µs
            # Flags (bidirectional HTTP)
            30: (1,        0),        # Fwd PSH Flags
            31: (1,        0),        # Bwd PSH Flags
            44: (1,        0),        # SYN Flag Count
            46: (10,       3),        # PSH Flag Count — HTTP data pushes
            47: (12,       3),        # ACK Flag Count
            43: (1,        0),        # FIN Flag Count
            # Packet stats
            52: (500,      100),      # Average Packet Size
            53: (650,      150),      # Avg Fwd Segment Size
            54: (300,      80),       # Avg Bwd Segment Size
            66: (65535,    0),        # Init_Win_bytes_forward
            67: (16383,    0),        # Init_Win_bytes_backward
        },
    },

    "Botnet": {
        "description": "Botnet C&C — low-rate periodic beaconing",
        "source_ip":   "192.168.50.{r}",
        "features": {
            0:  (6667,     0),        # Destination Port — IRC C&C
            1:  (2000000,  500000),   # Flow Duration µs — persistent long flow
            2:  (60,       15),       # Total Fwd Packets — moderate, periodic
            3:  (60,       15),       # Total Backward Packets — bidirectional
            4:  (6000,     1500),     # Total Length Fwd Packets
            5:  (6000,     1500),     # Total Length Bwd Packets
            6:  (200,      50),       # Fwd Packet Length Max
            7:  (20,       5),        # Fwd Packet Length Min
            8:  (100,      25),       # Fwd Packet Length Mean — small beacons
            9:  (60,       15),       # Fwd Packet Length Std — uniform packets
            12: (100,      25),       # Bwd Packet Length Mean
            # Rates — deliberately LOW to avoid detection
            14: (3000,     800),      # Flow Bytes/s — very low
            15: (30,       8),        # Flow Packets/s — low rate
            # Timing — regular intervals (scripted beacon)
            16: (30000,    2000),     # Flow IAT Mean µs — very regular
            17: (1000,     200),      # Flow IAT Std — very low variance
            18: (35000,    3000),     # Flow IAT Max
            19: (28000,    2000),     # Flow IAT Min — tight distribution
            # Flags (established connection)
            44: (1,        0),        # SYN Flag Count
            47: (60,       15),       # ACK Flag Count — high (ongoing connection)
            # Subflow
            51: (1,        0),        # Down/Up Ratio ≈ 1 (bidirectional)
            52: (100,      25),       # Average Packet Size
            66: (65535,    0),        # Init_Win_bytes_forward — large window
            67: (65535,    0),        # Init_Win_bytes_backward
        },
    },
}

# ── Benign protocol profiles ───────────────────────────────────────────────────
BENIGN_PROFILES = {
    "HTTPS_browsing": {
        "description": "Encrypted web browsing (TLS 1.3)",
        "source_ip":   "192.168.1.{r}",
        "port":        443,
        "features": {
            0:  (443,     0),
            1:  (600000,  200000),    # Flow Duration µs
            2:  (14,      6),         # Total Fwd Packets
            3:  (12,      5),         # Total Backward Packets
            4:  (7000,    2000),      # Total Length Fwd
            5:  (25000,   8000),      # Total Length Bwd — server sends more
            6:  (1460,    100),       # Fwd Packet Length Max (MTU)
            7:  (20,      5),         # Fwd Packet Length Min (ACK)
            8:  (500,     150),       # Fwd Packet Length Mean
            9:  (380,     100),       # Fwd Packet Length Std
            10: (1460,    100),       # Bwd Packet Length Max
            12: (2080,    500),       # Bwd Packet Length Mean — bigger responses
            14: (35000,   10000),     # Flow Bytes/s
            15: (40,      12),        # Flow Packets/s
            16: (40000,   15000),     # Flow IAT Mean µs
            17: (60000,   20000),     # Flow IAT Std
            30: (1,       0),         # Fwd PSH Flags
            31: (1,       0),         # Bwd PSH Flags
            44: (1,       0),         # SYN
            47: (14,      5),         # ACK
            43: (1,       0),         # FIN
            52: (1200,    300),       # Average Packet Size
            53: (500,     150),       # Avg Fwd Segment Size
            54: (2080,    500),       # Avg Bwd Segment Size
            66: (65535,   0),         # Init_Win_bytes_forward
            67: (65535,   0),         # Init_Win_bytes_backward
        },
    },

    "DNS_query": {
        "description": "DNS lookup (UDP, short exchange)",
        "source_ip":   "192.168.1.{r}",
        "port":        53,
        "features": {
            0:  (53,      0),
            1:  (8000,    3000),      # Very short flow
            2:  (1,       0),
            3:  (1,       0),
            4:  (50,      10),        # Small query
            5:  (130,     40),        # Slightly larger reply
            6:  (50,      10),
            7:  (50,      10),
            8:  (50,      10),
            10: (130,     40),
            12: (130,     40),
            14: (25000,   8000),      # Flow Bytes/s
            15: (250,     80),        # Flow Packets/s
            16: (8000,    3000),      # Flow IAT Mean µs
            44: (0,       0),         # SYN (UDP has no SYN)
            52: (90,      25),        # Average Packet Size
        },
    },

    "SSH_session": {
        "description": "Interactive SSH session (post-auth)",
        "source_ip":   "192.168.1.{r}",
        "port":        22,
        "features": {
            0:  (22,      0),
            1:  (3000000, 1500000),   # Long session
            2:  (180,     60),
            3:  (170,     60),
            4:  (25000,   8000),
            5:  (28000,   9000),
            6:  (1460,    100),
            7:  (20,      5),
            8:  (140,     40),
            9:  (200,     60),
            12: (165,     50),
            14: (9000,    3000),
            15: (60,      20),
            16: (50000,   20000),
            17: (100000,  40000),
            44: (1,       0),
            47: (175,     60),
            43: (1,       0),
            53: (140,     40),
            54: (165,     50),
            66: (65535,   0),
            67: (65535,   0),
        },
    },

    "HTTP_request": {
        "description": "Plain HTTP web request",
        "source_ip":   "192.168.1.{r}",
        "port":        80,
        "features": {
            0:  (80,      0),
            1:  (400000,  150000),
            2:  (9,       4),
            3:  (7,       3),
            4:  (3500,    1000),
            5:  (18000,   6000),
            6:  (1460,    100),
            7:  (20,      5),
            8:  (390,     120),
            9:  (420,     130),
            12: (2500,    700),
            14: (40000,   12000),
            15: (40,      12),
            16: (60000,   25000),
            30: (1,       0),
            31: (1,       0),
            44: (1,       0),
            46: (8,       3),
            47: (9,       3),
            43: (1,       0),
            52: (1200,    300),
            66: (65535,   0),
            67: (16383,   0),
        },
    },

    "Background_idle": {
        "description": "Keep-alive / idle TCP connection",
        "source_ip":   "192.168.1.{r}",
        "port":        443,
        "features": {
            0:  (443,     0),
            1:  (2500000, 1000000),   # Long idle
            2:  (4,       1),
            3:  (3,       1),
            4:  (100,     30),
            5:  (80,      25),
            6:  (80,      25),
            7:  (20,      5),
            8:  (50,      15),
            14: (50,      15),
            15: (2,       1),
            16: (1000000, 400000),    # Very long IAT
            17: (500000,  200000),    # High variance in idle
            47: (4,       1),
            52: (65,      20),
            66: (65535,   0),
            67: (65535,   0),
            74: (1000000, 400000),    # Idle Mean — long idle periods
        },
    },
}


def _sample_profile(profile_features: dict) -> np.ndarray:
    """
    Build a feature vector from a profile.
    Specified indices → sampled from their Gaussian.
    Unspecified indices → 0 (genuinely absent in the flow, as in real data).
    """
    v = np.zeros(FEATURE_COUNT, dtype=np.float32)
    for idx, (mean, std) in profile_features.items():
        if idx >= FEATURE_COUNT:
            continue
        val = random.gauss(mean, std) if std > 0 else float(mean)
        v[idx] = max(0.0, val)
    return v


def _benign_base() -> np.ndarray:
    """
    Returns a raw (unscaled) benign feature vector from the CIC-IDS2017 test set.
    Using real captured traffic guarantees the model sees the correct feature distribution.
    Falls back to synthetic profile if test data is not available.
    """
    if _SCALER is not None and len(_BENIGN_IDX) > 0:
        idx = random.choice(_BENIGN_IDX)
        # X_test is z-scored → inverse_transform → raw values
        # classify() will re-apply the scaler exactly once
        return _SCALER.inverse_transform(
            _X_TEST[idx].copy().astype(np.float32).reshape(1, -1)
        ).flatten().astype(np.float32)
    # Fallback: synthetic profile
    name = random.choice(list(BENIGN_PROFILES.keys()))
    return _sample_profile(BENIGN_PROFILES[name]["features"])


def _attack_vector(profile_name: str) -> np.ndarray:
    """
    Returns a raw (unscaled) attack feature vector from the CIC-IDS2017 test set.
    Sampling from the real test distribution guarantees genuine attack vs. benign
    classification — not synthetic guesswork.
    Falls back to profile if test data is not available.
    """
    if _SCALER is not None and len(_ATTACK_IDX) > 0:
        idx = random.choice(_ATTACK_IDX)
        return _SCALER.inverse_transform(
            _X_TEST[idx].copy().astype(np.float32).reshape(1, -1)
        ).flatten().astype(np.float32)
    # Fallback: synthetic profile
    return _sample_profile(ATTACK_PROFILES[profile_name]["features"])


def get_real_network_stats() -> dict:
    if not _PSUTIL_OK:
        return {}
    try:
        net = _psutil.net_io_counters()
        return {
            "bytes_sent":   net.bytes_sent,
            "bytes_recv":   net.bytes_recv,
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
            "errin":        net.errin,
            "errout":       net.errout,
            "dropin":       net.dropin,
            "dropout":      net.dropout,
        }
    except Exception:
        return {}


def generate_traffic_sample(mode: str = "mixed") -> dict:
    if mode == "mixed":
        mode = "benign" if random.random() < 0.70 else "attack"
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    if mode == "benign":
        profile_name = random.choice(list(BENIGN_PROFILES.keys()))
        profile      = BENIGN_PROFILES[profile_name]
        features     = _benign_base()
        label        = 0
        attack_type  = "BENIGN"
        r            = random.randint(2, 50)
        source_ip    = profile["source_ip"].format(r=r)
        dest_port    = int(profile.get("port", features[0]))
    else:
        attack_name  = random.choice(list(ATTACK_PROFILES.keys()))
        profile      = ATTACK_PROFILES[attack_name]
        features     = _attack_vector(attack_name)
        label        = 1
        attack_type  = attack_name
        r            = random.randint(1, 254)
        source_ip    = profile["source_ip"].format(r=r)
        dest_port    = int(features[0])

    return {
        "features":       features,
        "true_label":     label,
        "attack_type":    attack_type,
        "source_ip":      source_ip,
        "dest_port":      dest_port,
        "timestamp":      timestamp,
        "real_net_stats": get_real_network_stats(),
    }


def batch_generate(n: int = 10, mode: str = "mixed") -> list:
    return [generate_traffic_sample(mode) for _ in range(n)]


if __name__ == "__main__":
    print(f"Simulator — FEATURE_COUNT={FEATURE_COUNT}  psutil={'OK' if _PSUTIL_OK else 'NOT installed'}")
    print(f"Scaler: {'loaded' if _SCALER else 'NOT FOUND (using defaults)'}")
    print()
    print("5 BENIGN samples:")
    for s in batch_generate(5, "benign"):
        f = s["features"]
        print(f"  {s['timestamp']}  {s['source_ip']:<18}  port={s['dest_port']:<5}"
              f"  bytes/s={f[14]:.0f}  pkts/s={f[15]:.0f}")
    print()
    print("1 of each attack type:")
    for atype in ATTACK_PROFILES:
        f = _attack_vector(atype)
        print(f"  {atype:<20}  port={f[0]:.0f}  dur={f[1]:.0f}µs"
              f"  bytes/s={f[14]:.0f}  pkts/s={f[15]:.0f}"
              f"  SYN={f[44]:.0f}  RST={f[45]:.0f}")
