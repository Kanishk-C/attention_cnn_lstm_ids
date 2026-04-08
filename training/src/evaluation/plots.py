"""
src/evaluation/plots.py  —  All matplotlib plot functions.

matplotlib.use("Agg") must be called before pyplot import.
This makes plots work on headless servers, Docker, SSH sessions,
and any machine without a display — including Google Colab workers.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODEL_COLORS = {
    "CNN":                "#378ADD",
    "LSTM":               "#1D9E75",
    "CNN-LSTM":           "#D85A30",
    "Attention-CNN-LSTM": "#7F77DD",
}
_DEFAULT = "#888780"


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [plot] → {path}")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["BENIGN", "ATTACK"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, pad=10)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_loss_curves(histories: dict, save_path=None):
    """histories: {model_name: [loss_per_epoch, ...]}"""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, losses in histories.items():
        ax.plot(range(1, len(losses) + 1), losses,
                label=name, color=MODEL_COLORS.get(name, _DEFAULT),
                linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss per Epoch", fontsize=12, pad=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(1, max(len(v) for v in histories.values()) + 1))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_comparison_bar(results: dict, save_path=None):
    """results: {model_name: {accuracy, f1, mcc}}"""
    metrics = ["accuracy", "f1", "mcc"]
    labels  = ["Accuracy", "F1-Score", "MCC"]
    names   = list(results.keys())
    x       = np.arange(len(metrics))
    n       = len(names)
    width   = 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        vals = [results[name].get(m, 0) for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=name, color=MODEL_COLORS.get(name, _DEFAULT), alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
    ax.set_title("Model Comparison — Accuracy / F1 / MCC", fontsize=12, pad=10)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(top=0.88, bottom=0.12)
    _save(fig, save_path)
    return fig


def plot_radar(results: dict, save_path=None):
    import math
    cats   = ["Accuracy", "F1-Score", "MCC"]
    N      = len(cats)
    angles = [n / float(N) * 2 * math.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    for name, m in results.items():
        vals = [m.get("accuracy", 0), m.get("f1", 0), m.get("mcc", 0)]
        vals += vals[:1]
        ax.plot(angles, vals, color=MODEL_COLORS.get(name, _DEFAULT),
                linewidth=2, label=name)
        ax.fill(angles, vals, color=MODEL_COLORS.get(name, _DEFAULT), alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), cats)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title("Performance Radar", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
