# 💻 OS-Specific Installation & Setup Guide

This guide provides detailed instructions for setting up the **Attention-CNN-LSTM IDS** on Linux, Windows, and macOS.

---

## 🐧 Linux (Recommended & Best Performance)

Linux is the native environment for network packet capture because it allows `nfstream` to hook directly into the kernel's high-performance network stack using **eBPF (Extended Berkeley Packet Filter)** or the raw **AF_PACKET** socket interface. This bypasses user-space overhead, allowing for true real-time, Gigabit-speed analysis without dropping packets.

### 1. System Requirements & Libraries
To compile the underlying C bindings for `nfstream`, you require the Linux packet capture library (`libpcap-dev`).
```bash
sudo apt update
sudo apt install libpcap-dev python3-dev build-essential
```

### 2. Virtual Environment Setup
Avoid installing dependencies globally to prevent system tool conflicts.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Execution & Permissions Boundary
Linux strictly isolates raw socket promiscuous mode access to the `root` user (`sudo` or `CAP_NET_RAW` capability). 
Because you are using a virtual environment, you must explicitly point `sudo` to the virtual environment's Python executable.
```bash
# Capture traffic on eth0
sudo .venv/bin/python app/capture_agent.py --interface eth0
```

---

## 🪟 Windows Setup

Windows fundamentally differs from Unix-like systems. It manages networking through the **NDIS (Network Driver Interface Specification)** stack. Python cannot natively read raw sockets from NDIS. Therefore, `nfstream` on Windows depends entirely on an injected kernel-mode driver known as **Npcap**.

### 1. Driver Installation (Npcap)
1. Download Npcap from [npcap.com](https://npcap.com/).
2. Run the installer. **CRITICAL STEP:** During installation, you must tick the box that says **"Install Npcap in WinPcap API-compatible Mode"**. If you skip this, `nfstream` cannot hook into the driver.

### 2. Virtual Environment Setup
Run this in PowerShell or Command Prompt. Ensure you use the native `python` command if it is registered in your PATH.
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Execution & Permissions Boundary
Windows implements User Account Control (UAC). To interact with the Npcap driver, your execution context must have administrative rights.
1. Right-click your terminal application (PowerShell, CMD, or VSCode).
2. Select **"Run as Administrator"**.
3. Use the `--list-interfaces` flag to find the Windows-formatted interface name (e.g., "Wi-Fi" or "Ethernet"), then launch the agent:
```powershell
python app/capture_agent.py --interface "Wi-Fi"
```

---

## 🍎 macOS Setup

macOS (built on Darwin BSD) utilizes the **BPF (Berkeley Packet Filter)** device interface at `/dev/bpf*`. Similar to Linux, you need a C-level packet library to interface with BPF.

### 1. Core Libraries via Homebrew
Ensure you have Homebrew installed, then grab the `libpcap` package.
```bash
brew install libpcap
```

### 2. Virtual Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Execution & Permissions Boundary
macOS strictly restricts `/dev/bpf` devices to the `root` user. You must use `sudo` combined with your virtual environment's python path.
```bash
# Find interface (commonly en0 for Wi-Fi on Mac)
sudo .venv/bin/python app/capture_agent.py --list-interfaces

# Start capture agent
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
