# Attention-CNN-LSTM Intrusion Detection System

## Overview
This project implements and evaluates deep learning models for network intrusion detection using the CIC-IDS2017 dataset. The study compares CNN, LSTM, CNN–LSTM, and Attention-based CNN–LSTM architectures under CPU-only constraints.

## Dataset
- CIC-IDS2017 (flow-based features)
- Binary classification: BENIGN vs ATTACK
- Data preprocessing includes normalization, label encoding, and leakage removal

## Models Implemented
- CNN baseline
- LSTM baseline
- CNN–LSTM hybrid
- Attention-CNN-LSTM (final model)

## Results Summary
| Model | Accuracy | F1 | MCC |
|------|---------|----|-----|
| CNN | 0.9957 | 0.9903 | 0.9875 |
| LSTM | 0.9897 | 0.9769 | 0.9704 |
| CNN–LSTM | 0.9926 | 0.9836 | 0.9788 |
| Attention-CNN-LSTM | 0.9937 | 0.9861 | 0.9821 |

## Environment
- Python 3.10
- CPU-only execution
- PyTorch

## Notes
This project focuses on model development and evaluation. Real-time deployment and system integration are considered future work.

