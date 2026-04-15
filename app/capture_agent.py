import os
import sys
import argparse
import fcntl
import csv


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

# Priority 1: Shared Memory (RAM) for SSD Longevity
if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
    QUEUE_FILE = "/dev/shm/ids_traffic_queue.csv"
else:
    QUEUE_FILE = os.path.join(os.path.dirname(__file__), "data", "captures", ".traffic_queue.csv")
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)




def map_nfstream_to_cicids(flow):
    """Approximates the 78 CIC-IDS2017 features from nfstream output."""
    
    features = [0.0] * 78
    
    # 1. Basic Flow Info
    features[0] = flow.dst_port                  # Destination Port

    features[1] = flow.bidirectional_duration_ms  # Flow Duration
    features[2] = flow.src2dst_packets           # Total Fwd Packets
    features[3] = flow.dst2src_packets           # Total Backward Packets
    features[4] = flow.src2dst_bytes             # Total Length of Fwd Packets
    features[5] = flow.dst2src_bytes             # Total Length of Bwd Packets
    
    # 2. Fwd/Bwd Packet Length Stats (6-13)
    features[6] = getattr(flow, 'src2dst_max_ps', 0)
    features[7] = getattr(flow, 'src2dst_min_ps', 0)
    features[8] = getattr(flow, 'src2dst_mean_ps', 0)
    features[9] = getattr(flow, 'src2dst_stddev_ps', 0)
    
    features[10] = getattr(flow, 'dst2src_max_ps', 0)
    features[11] = getattr(flow, 'dst2src_min_ps', 0)
    features[12] = getattr(flow, 'dst2src_mean_ps', 0)
    features[13] = getattr(flow, 'dst2src_stddev_ps', 0)

    # 3. Flow Bytes/Packets per sec (14-15)
    duration_s = max(flow.bidirectional_duration_ms / 1000.0, 0.0001)
    features[14] = flow.bidirectional_bytes / duration_s
    features[15] = flow.bidirectional_packets / duration_s

    # 4. Flow IAT Stats (16-19)
    features[16] = getattr(flow, 'bidirectional_mean_piat_ms', 0)
    features[17] = getattr(flow, 'bidirectional_stddev_piat_ms', 0)
    features[18] = getattr(flow, 'bidirectional_max_piat_ms', 0)
    features[19] = getattr(flow, 'bidirectional_min_piat_ms', 0)

    # 5. TCP Flags (approximate from packet counts)
    features[44] = 1 if getattr(flow, 'bidirectional_fin_packets', 0) > 0 else 0
    features[45] = 1 if getattr(flow, 'bidirectional_syn_packets', 0) > 0 else 0
    features[46] = 1 if getattr(flow, 'bidirectional_rst_packets', 0) > 0 else 0
    features[47] = 1 if getattr(flow, 'bidirectional_psh_packets', 0) > 0 else 0
    features[48] = 1 if getattr(flow, 'bidirectional_ack_packets', 0) > 0 else 0

    # 6. Overall Packet Stats (38-42)
    features[38] = getattr(flow, 'bidirectional_min_ps', 0)
    features[39] = getattr(flow, 'bidirectional_max_ps', 0)
    features[40] = getattr(flow, 'bidirectional_mean_ps', 0)
    features[41] = getattr(flow, 'bidirectional_stddev_ps', 0)

    
    return features

def safe_append_to_queue(src_ip, dst_ip, src_port, dst_port, features_list):
    """Appends flow data to .traffic_queue.csv safely using file locking."""
    # Ensure file exists and has a header
    file_exists = os.path.exists(QUEUE_FILE)
    
    header = ["src_ip", "dst_ip", "src_port", "dst_port"] + [f"feat_{i}" for i in range(78)]
    
    with open(QUEUE_FILE, 'a+', newline='') as f:

        # Acquire an exclusive lock
        if sys.platform != 'win32':
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
        try:
            writer = csv.writer(f)
            
            # Rolling buffer: Parse lines if file exceeds ~20MB
            if file_exists and os.path.getsize(QUEUE_FILE) > 20 * 1024 * 1024:
                f.seek(0)
                lines = f.readlines()
                if len(lines) > 10000:
                    # Keep header + last 9000 lines to avoid immediate re-truncation
                    new_content = [lines[0]] + lines[-9000:]
                    f.seek(0)
                    f.truncate()
                    f.writelines(new_content)
                    f.seek(0, os.SEEK_END)

            
            if not file_exists or os.path.getsize(QUEUE_FILE) == 0:
                writer.writerow(header)
                
            # Row format: metadata + 78 features
            row = [src_ip, dst_ip, src_port, dst_port] + features_list
            writer.writerow(row)

            

            
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
        streamer = NFStreamer(source=interface,  
                              active_timeout=5, 
                              idle_timeout=5, 
                              statistical_analysis=True)



        for flow in streamer:
            # Map raw flow output to CICIDS2017 features
            mapped_features = map_nfstream_to_cicids(flow)
            
            if dry_run:
                print(f"[Flow] {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} | Packets: {flow.bidirectional_packets}")
            else:
                safe_append_to_queue(
                    flow.src_ip, flow.dst_ip, 
                    getattr(flow, 'src_port', 0), getattr(flow, 'dst_port', 0),
                    mapped_features
                )

                
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

def list_interfaces():
    """Lists available network interfaces using psutil."""
    import psutil
    print("\n[*] Available Network Interfaces:")
    addrs = psutil.net_if_addrs()
    for name, _ in addrs.items():
        print(f"  - {name}")
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live traffic capture agent for Streamlit IDS")
    parser.add_argument("--interface", type=str, default="eth0", help="Network interface to monitor")
    parser.add_argument("--dry-run", action="store_true", help="Print features to console instead of sending to dashboard")
    parser.add_argument("--list-interfaces", action="store_true", help="List available network interfaces and exit")
    args = parser.parse_args()
    
    if args.list_interfaces:
        list_interfaces()
        sys.exit(0)
        
    start_capture(args.interface, args.dry_run)

