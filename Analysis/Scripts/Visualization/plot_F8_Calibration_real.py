import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
OUT_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
os.makedirs(OUT_DIR, exist_ok=True)
STAGE_A_OUT = os.path.join(ROOT, "Analysis", "Output", "Track0_Predictive", "stage_a_hazard_results.csv")

def plot_f8():
    print("==============================================")
    print(" Rendering Authentic F8: Calibration & Gains")
    print("==============================================")
    
    if not os.path.exists(STAGE_A_OUT):
        print("[-] Required Stage A predictive data not found.")
        return
        
    df = pd.read_csv(STAGE_A_OUT, usecols=['event_next_1yr', 'Prob_LR_H=4', 'Prob_H=4'])
    y_true = df['event_next_1yr']
    
    # Calibration Curves
    prob_true_c, prob_pred_c = calibration_curve(y_true, df['Prob_H=4'], n_bins=10)
    prob_true_b, prob_pred_b = calibration_curve(y_true, df['Prob_LR_H=4'], n_bins=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Reliability Diagram
    ax1.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')
    ax1.plot(prob_pred_c, prob_true_c, marker='o', linewidth=2, color='darkred', label='CatBoost (V-REx)')
    ax1.plot(prob_pred_b, prob_true_b, marker='s', linestyle=':', color='gray', label='Logistic Baseline')
    ax1.set_title("Panel A: Reliability Diagram")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Top-K Capture (Gains)
    # Sort by probability descending
    df_sorted_c = df.sort_values('Prob_H=4', ascending=False).reset_index(drop=True)
    df_sorted_c['cumulative_events'] = df_sorted_c['event_next_1yr'].cumsum()
    
    df_sorted_b = df.sort_values('Prob_LR_H=4', ascending=False).reset_index(drop=True)
    df_sorted_b['cumulative_events'] = df_sorted_b['event_next_1yr'].cumsum()
    
    total_events = y_true.sum()
    percentiles = np.linspace(0, 100, len(df))
    
    capture_c = (df_sorted_c['cumulative_events'] / total_events) * 100
    capture_b = (df_sorted_b['cumulative_events'] / total_events) * 100
    
    ax2.plot(percentiles, capture_c, linewidth=2, color='darkred', label='CatBoost Capture')
    ax2.plot(percentiles, capture_b, linestyle='--', color='gray', label='Logistic Capture')
    ax2.plot([0, 100], [0, 100], linestyle=':', color='black', label='Random Baseline')
    
    ax2.set_title("Panel B: Top-K Capture (Gains Rate)")
    ax2.set_xlabel("Top Percentile of Ranked Sites")
    ax2.set_ylabel("Percentage of Realized Events Captured")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Exhibit F8: Empirical Development Calibration and Information Gains', fontsize=14, y=1.05)
    plt.tight_layout()
    f8_path = os.path.join(OUT_DIR, "F8_Calibration.png")
    plt.savefig(f8_path, dpi=300, bbox_inches='tight')
    print(f"[+] Successfully saved {f8_path}")

if __name__ == '__main__':
    plot_f8()
