import os
import sys
import json
import time
import argparse
import fcntl

# Cross-OS Compatibility check for nfstream
try:
    from nfstream import NFStreamer
except ImportError:
    print("CRITICAL: nfstream is not installed.")
    print("Please run: pip install nfstream")
    sys.exit(1)

def print_os_help():
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        print("\nNote: On Linux/macOS, live packet sniffing requires superuser privileges.")
        print("Please restart this script using: sudo python app/capture_agent.py --interface INTERFACE_NAME\n")
    elif sys.platform == 'win32':
        print("\nNote: On Windows, live packet sniffing requires Npcap (https://npcap.com/) and Administrator privileges.")
        print("Please ensure Npcap is installed and run your terminal/Command Prompt as Administrator.\n")

QUEUE_FILE = os.path.join(os.path.dirname(__file__), ".traffic_queue.json")

def map_nfstream_to_cicids(flow):
    """
    Approximates the 78 CIC-IDS2017 features from nfstream output.
    Note: nfstream and CICFlowMeter have slight differences in statistical aggregations, 
    so this assigns closely related features.
    """
    # This is a simplified mapper. In a fully robust deployment, we would map all 78 perfectly.
    # To prevent dashboard crash due to mismatch, we pad the array up to 78 numeric fields.
    # We will prioritize capturing length, byte stats, duration, etc.
    
    # We will use dummy mapping for the missing exact CICFlowMeter metrics
    # while transferring key nfstream fields.
    features = [0.0] * 78 
    
    # Example mapping (indexes as per `feature_names.txt`):
    features[0] = flow.dst_port                 # Destination Port
    features[1] = flow.bidirectional_duration_ms # Flow Duration
    features[2] = flow.src2dst_packets          # Total Fwd Packets
    features[3] = flow.dst2src_packets          # Total Backward Packets
    features[4] = flow.src2dst_bytes            # Total Length of Fwd Packets
    features[5] = flow.dst2src_bytes            # Total Length of Bwd Packets
    # Fwd Packet max/min/mean/std (6-9)
    features[8] = flow.src2dst_bytes / max(flow.src2dst_packets, 1) # Fwd Packet Length Mean
    # Bwd Packet max/min/mean/std (10-13)
    features[12] = flow.dst2src_bytes / max(flow.dst2src_packets, 1) # Bwd Packet Length Mean
    
    # Calculate bytes/s and packets/s
    duration_s = max(flow.bidirectional_duration_ms / 1000.0, 0.0001)
    features[14] = flow.bidirectional_bytes / duration_s   # Flow Bytes/s
    features[15] = flow.bidirectional_packets / duration_s # Flow Packets/s
    
    # TCP flags (44-51 approx)
    features[44] = getattr(flow, 'bidirectional_fin_packets', 0)
    features[45] = getattr(flow, 'bidirectional_syn_packets', 0)
    features[46] = getattr(flow, 'bidirectional_rst_packets', 0)
    features[47] = getattr(flow, 'bidirectional_psh_packets', 0)
    features[48] = getattr(flow, 'bidirectional_ack_packets', 0)
    
    return features

def safe_append_to_queue(features_list):
    """Appends to .traffic_queue.json safely using file locking."""
    # Ensure file exists
    if not os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, 'w') as f:
            json.dump([], f)
            
    with open(QUEUE_FILE, 'r+') as f:
        # Acquire an exclusive lock
        if sys.platform != 'win32':
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
        try:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
                
            # Keep queue size manageable (last 1000 items)
            data.append(features_list)
            if len(data) > 1000:
                data = data[-1000:]
                
            f.seek(0)
            f.truncate()
            json.dump(data, f)
            
        finally:
            if sys.platform != 'win32':
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def start_capture(interface, dry_run=False):
    print(f"[*] Starting NFStreamer on interface: {interface}")
    if not dry_run:
        print(f"[*] Live traffic will be written to {QUEUE_FILE}")
    else:
        print("[*] DRY RUN MODE: Traffice will be printed, not forwarded.")
        
    try:
        streamer = NFStreamer(source=interface, active_timeout=10, idle_timeout=30)
        for flow in streamer:
            # Map raw flow output to CICIDS2017 features
            mapped_features = map_nfstream_to_cicids(flow)
            
            if dry_run:
                print(f"[Flow] {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} | Packets: {flow.bidirectional_packets}")
            else:
                safe_append_to_queue(mapped_features)
                
    except PermissionError:
        print("\n[!] PERMISSION DENIED.")
        print_os_help()
    except ValueError as e:
        if "Unknown Network Interface" in str(e) or "No such device" in str(e):
             print(f"\n[!] Interface '{interface}' not found. Use 'ifconfig' or 'ipconfig' to list available interfaces.")
        else:
             print(f"\n[!] NFStreamer Error: {e}")
    except KeyboardInterrupt:
        print("\n[*] Capture stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live traffic capture agent for Streamlit IDS")
    parser.add_argument("--interface", type=str, default="eth0", help="Network interface to monitor")
    parser.add_argument("--dry-run", action="store_true", help="Print features to console instead of sending to dashboard")
    args = parser.parse_args()
    
    start_capture(args.interface, args.dry_run)
