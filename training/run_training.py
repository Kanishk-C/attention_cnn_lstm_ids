"""
run_all.py  —  Train all four IDS models in one command.

The  if __name__ == "__main__":  guard is REQUIRED on Windows.
Without it, DataLoader worker subprocesses re-import this file
and try to start training again, causing an infinite crash loop.

Usage:
    python run_all.py                          # full 10-epoch run
    python run_all.py --epochs 3               # quick test
    python run_all.py --model CNN              # one model only
    python run_all.py --epochs 3 --model CNN   # fastest smoke-test
"""

if __name__ == "__main__":

    import os
    import sys
    import json
    import time
    import argparse
    import torch
    import pandas as pd

    from src.models.cnn_baseline       import CNNBaseline
    from src.models.lstm_baseline      import LSTMBaseline
    from src.models.cnn_lstm           import CNNLSTM
    from src.models.attention_cnn_lstm import AttentionCNNLSTM
    from src.utils.train_utils         import load_data, train_model, evaluate_model
    from src.evaluation.plots          import (
        plot_confusion_matrix,
        plot_loss_curves,
        plot_comparison_bar,
        plot_radar,
    )

    CKPT_DIR = os.path.join("experiments", "checkpoints")
    LOG_DIR  = "logs"

    ALL_MODELS = {
        "CNN":                CNNBaseline,
        "LSTM":               LSTMBaseline,
        "CNN-LSTM":           CNNLSTM,
        "Attention-CNN-LSTM": AttentionCNNLSTM,
    }

    # ── CLI args ──────────────────────────────────────────────────
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--model", choices=list(ALL_MODELS.keys()), default=None,
                   help="Train one model only (default: all four)")
    args = p.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print("=" * 58)
    print("  Intrusion Detection System — Training")
    print("=" * 58)
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Platform   : {sys.platform}  |  Cores: {os.cpu_count() or '?'}")

    X_train, y_train, X_test, y_test = load_data()
    NF = X_train.shape[1]
    print(f"  Features   : {NF}")
    print(f"  Train rows : {X_train.shape[0]:,}")
    print(f"  Test rows  : {X_test.shape[0]:,}\n")

    to_run = {args.model: ALL_MODELS[args.model]} if args.model else ALL_MODELS

    all_results   = {}
    all_histories = {}
    t_total = time.time()

    for name, Cls in to_run.items():
        print(f"\n{'─'*58}")
        print(f"  Training: {name}")
        print(f"{'─'*58}")

        model = Cls(NF)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters : {n_params:,}")

        t0 = time.time()
        history = train_model(model, X_train, y_train,
                              epochs=args.epochs,
                              batch_size=args.batch_size,
                              lr=args.lr)
        elapsed = time.time() - t0
        print(f"  Time : {elapsed:.1f}s ({elapsed/args.epochs:.1f}s/epoch)")

        print(f"\n  Evaluating: {name}")
        metrics, preds, y_true = evaluate_model(model, X_test, y_test)
        metrics["train_time_s"] = round(elapsed, 1)
        metrics["n_params"]     = n_params

        all_results[name]   = metrics
        all_histories[name] = history["loss"]

        # Save checkpoint — always on CPU so it loads anywhere
        ckpt = os.path.join(CKPT_DIR, f"{name.replace(' ', '_')}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"  Checkpoint → {ckpt}")

        plot_confusion_matrix(
            y_true, preds, name,
            save_path=os.path.join(LOG_DIR, f"cm_{name.replace(' ', '_')}.png")
        )

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("  RESULTS SUMMARY")
    print(f"{'='*58}")
    fmt = "  {:<25}  {:>6}  {:>6}  {:>6}  {:>8}"
    print(fmt.format("Model", "Acc", "F1", "MCC", "Time"))
    print("  " + "─" * 52)

    rows = []
    for name, m in all_results.items():
        print(fmt.format(name,
                         f"{m['accuracy']:.4f}",
                         f"{m['f1']:.4f}",
                         f"{m['mcc']:.4f}",
                         f"{m['train_time_s']}s"))
        rows.append({"Model": name,
                     "Accuracy": round(m["accuracy"], 4),
                     "F1":       round(m["f1"],       4),
                     "MCC":      round(m["mcc"],      4),
                     "Params":   m["n_params"],
                     "Time (s)": m["train_time_s"]})

    pd.DataFrame(rows).to_csv(
        os.path.join(LOG_DIR, "results.csv"), index=False, encoding="utf-8"
    )

    if len(all_results) > 1:
        plot_loss_curves(all_histories,
                         save_path=os.path.join(LOG_DIR, "loss_curves.png"))
        plot_comparison_bar(all_results,
                            save_path=os.path.join(LOG_DIR, "comparison_bar.png"))
        plot_radar(all_results,
                   save_path=os.path.join(LOG_DIR, "radar.png"))

    with open(os.path.join(LOG_DIR, "histories.json"), "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)

    total = time.time() - t_total
    print(f"\n  Total time : {total/60:.1f} min")
    print(f"  Results    → {LOG_DIR}/results.csv")
    print(f"  Plots      → {LOG_DIR}/")
    print(f"  Checkpoints→ {CKPT_DIR}/")
    print("\n[DONE]  Run:  streamlit run app.py")
