"""
compare_runs.py
===============
Aggregates results CSVs across training runs into a single
cross-run comparison table for hyperparameter evaluation.
"""
import os
import glob
import pandas as pd

THESIS_DIR = r"C:\Users\dhl\data\Thesis\thesis"
OUT_DIR    = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

# ── 1. LSTM horizon results ──────────────────────────────────
lstm_csvs = glob.glob(os.path.join(THESIS_DIR, "dynamic_lstm_horizon_results*.csv"))
lstm_frames = []
for path in lstm_csvs:
    run_name = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path)
    df["Run"] = run_name
    lstm_frames.append(df)

if lstm_frames:
    lstm_combined = pd.concat(lstm_frames, ignore_index=True)
    lstm_combined = lstm_combined.sort_values(["Horizon", "Run"])
    lstm_out = os.path.join(OUT_DIR, "cross_run_lstm_comparison.csv")
    lstm_combined.to_csv(lstm_out, index=False)
    print("=== LSTM Cross-Run Comparison ===")
    print(lstm_combined.to_string(index=False))
    print(f"\nSaved to {lstm_out}")
else:
    print("No LSTM results CSVs found yet.")

# ── 2. Training logs summary ─────────────────────────────────
log_files = {
    "v3": os.path.join(THESIS_DIR, "model_retraining_logs_v3.txt"),
    "v4": os.path.join(THESIS_DIR, "model_retraining_logs_v4.txt"),
    "v5": os.path.join(THESIS_DIR, "model_retraining_logs_v5.txt"),
}

log_rows = []
for run, path in log_files.items():
    if not os.path.exists(path):
        continue
    with open(path, "r", errors="ignore") as f:
        content = f.read()
    
    # Parse epoch lines
    import re
    epoch_lines = re.findall(
        r"Epoch (\d+)/\d+.*?Train Loss: ([\d.]+).*?Val Loss: ([\d.]+).*?Val PR AUC: ([\d.]+)",
        content
    )
    horizon_lines = re.findall(r"Executing Horizon: (\S+)", content)
    pr_auc_lines  = re.findall(r"LSTM PR AUC: ([\d.]+)", content)
    best_epoch_lines = re.findall(r"\[Early Stop\] Best checkpoint: Epoch (\d+)", content)
    
    for i, (horizon, pr_auc) in enumerate(zip(horizon_lines, pr_auc_lines)):
        log_rows.append({
            "Run": run,
            "Horizon": horizon,
            "Final_PR_AUC": float(pr_auc),
            "Best_Epoch": best_epoch_lines[i] if i < len(best_epoch_lines) else "N/A",
        })

if log_rows:
    log_df = pd.DataFrame(log_rows)
    log_out = os.path.join(OUT_DIR, "cross_run_log_summary.csv")
    log_df.to_csv(log_out, index=False)
    print("\n=== Cross-Run Log Summary ===")
    print(log_df.pivot_table(index="Horizon", columns="Run", values="Final_PR_AUC").to_string())
    print(f"\nSaved to {log_out}")
else:
    print("No parseable log lines found yet.")

# ── 3. Ablation results ───────────────────────────────────────
ablation_path = os.path.join(OUT_DIR, "spatial_ablation_results.csv")
if os.path.exists(ablation_path):
    abl = pd.read_csv(ablation_path)
    print("\n=== Spatial Feature Ablation ===")
    print(abl.to_string(index=False))
else:
    print("\nAblation results not yet generated — run run_spatial_ablation.py first.")
