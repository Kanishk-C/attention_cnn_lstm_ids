# 🚀 Network Attack Simulation Guide

This guide provides three methods to simulate network attacks safely on your lab environment to test the **Attention-CNN-LSTM IDS** model.

> [!WARNING]
> **Legal Notice**: Only perform these tests on network interfaces and machines that you own or have explicit permission to test. Unauthorized penetration testing is illegal.

---

## 1. Port Scanning (Reconnaissance)
Port scanning is often the first step in an attack. The model detects it by identifying multiple connection attempts to different ports within a short timeframe.

### Using Nmap
`nmap` is the industry-standard tool for network exploration.
*   **Command**:
    ```bash
    # Perform a stealthy SYN scan
    sudo nmap -sS -T4 <target-ip>
    ```
*   **What the IDS sees**:
    - High number of forward packets per second.
    - Zero payloads (handshake attempts only).
    - Varying destination ports with a single common source IP.
*   **Expected Dashboard Prediction**: `🚨 ATTACK` (PortScan)

---

## 2. Denial of Service (DoS/DDoS)
DoS attacks aim to overwhelm a service. The model identifies this by looking for extreme packet frequencies and very low inter-arrival times.

### Using Hping3
`hping3` is a powerful tool to generate TCP/IP packets.
*   **Command**:
    ```bash
    # Flood target port 80 with SYN packets
    sudo hping3 -S --flood -V -p 80 <target-ip>
    ```
*   **What the IDS sees**:
    - `Flow Packets/s` and `Flow Bytes/s` spike to extreme levels.
    - `Flow IAT Mean` (Inter-Arrival Time) drops to almost zero.
    - Mostly unidirectional traffic (if the target is overwhelmed).
*   **Expected Dashboard Prediction**: `🚨 ATTACK` (DDoS/DoS)

---

## 3. Manual Feature Spoofing (CSV Method)
If you cannot run live tools, you can manually "trigger" the model by injecting simulated malicious statistics into a traffic file.

### Steps:
1.  Navigate to [sample_traffic.csv](file:///home/kxjxmdk/Dev/projects/attention_cnn_lstm_ids/app/data/sample_traffic.csv).
2.  Duplicate a row and change the following values to simulate an attack:
    - **Destination Port**: `22` (SSH), `23` (Telnet), or `445` (SMB).
    - **Flow Packets/s**: Set to a very high value (e.g., `1000000`).
    - **Total Length of Fwd Packets**: Set to a high value (e.g., `500000`).
3.  Save the file as `my_attack.csv`.
4.  In the Dashboard sidebar, use the **File Uploader** to upload `my_attack.csv`.
5.  Set **Traffic Mode** to `Simulation (CSV)` and press **▶ Start Monitoring**.

---

## Summary of Detection Logic
| Attack Type | Key Detecting Features |
| :--- | :--- |
| **PortScan** | Destination Port variety, Packet Length (Avg=0), Fwd Packets/s |
| **DoS/DDoS** | Flow Packets/s (Max), Flow IAT (Min), Flow Duration (Low) |
| **Brute Force** | Protocol (SSH/FTP), Bidirectional Flow Duration, Packet sizes |
