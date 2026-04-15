# 💻 OS-Specific Installation & Setup Guide

This guide provides detailed instructions for setting up the **Attention-CNN-LSTM IDS** on Linux, Windows, and macOS.

---

## 🐧 Linux (Recommended)
Linux provides the best performance for raw packet capture.

### 1. Prerequisites
You need `libpcap` for `nfstream` to work.
```bash
sudo apt update
sudo apt install libpcap-dev python3-dev
```

### 2. Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Permissions
Live capture requires root privileges to access the network interface:
```bash
sudo .venv/bin/python app/capture_agent.py --interface eth0
```

---

## 🪟 Windows
Windows setup requires extra steps for raw packet capture.

### 1. Install Npcap
`nfstream` on Windows depends on **Npcap**.
1.  Download Npcap from [npcap.com](https://npcap.com/).
2.  During installation, ensure the **"Install Npcap in WinPcap API-compatible Mode"** checkbox is **checked**.

### 2. Virtual Environment & Dependencies
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Permissions
Open your terminal (PowerShell or CMD) **as Administrator** to run the capture agent.
```powershell
python app/capture_agent.py --interface "Wi-Fi"
```
*Note: Use `python app/capture_agent.py --list-interfaces` to find your Windows interface name.*

---

## 🍎 macOS
macOS requires similar packet capture libraries to Linux.

### 1. Install Homebrew & Libpcap
```bash
brew install libpcap
```

### 2. Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Permissions
You may need to run the capture agent with `sudo`:
```bash
sudo .venv/bin/python app/capture_agent.py --interface en0
```

---

## 🛠️ Common Troubleshooting

### `ModuleNotFoundError: No module named 'nfstream'`
Ensure you ran `pip install -r requirements.txt`. On Linux, if it still fails, check if `libpcap-dev` is installed.

### `Permission Denied` during capture
Raw socket access is a restricted privilege. Always use `sudo` (Linux/macOS) or an Administrator terminal (Windows).

### Interface not found
Run the following to list available names on your specific system:
```bash
python app/capture_agent.py --list-interfaces
```
