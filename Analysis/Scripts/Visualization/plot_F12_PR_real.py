import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

from sklearn.metrics import precision_recall_curve, average_precision_score
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
OUT_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
os.makedirs(OUT_DIR, exist_ok=True)

# Try loading empirical output from Stage C or fallback if not fully arrayed to Stage A format
# Stage A hazard results contain `Prob_LGBM_H=4`, `Prob_LR_H=4` etc.
STAGE_A_OUT = os.path.join(ROOT, "Analysis", "Output", "Track0_Predictive", "stage_a_hazard_results.csv")

def plot_f12():
    print("==============================================")
    print(" Rendering Authentic F12: PR Curves")
    print("==============================================")
    
    if not os.path.exists(STAGE_A_OUT):
        print("[-] Required Stage A predictive data not found.")
        return
        
    df = pd.read_csv(STAGE_A_OUT, usecols=['event_next_1yr', 'Prob_LR_H=4', 'Prob_LGBM_H=4', 'Prob_H=4'])
    
    # We evaluate for the H=4 (1 Yr) horizon
    y_true = df['event_next_1yr']
    
    # Calculate PR Curves for the three empirical models
    p_lr, r_lr, _ = precision_recall_curve(y_true, df['Prob_LR_H=4'])
    auc_lr = average_precision_score(y_true, df['Prob_LR_H=4'])
    
    p_lgbm, r_lgbm, _ = precision_recall_curve(y_true, df['Prob_LGBM_H=4'])
    auc_lgbm = average_precision_score(y_true, df['Prob_LGBM_H=4'])
    
    p_cb, r_cb, _ = precision_recall_curve(y_true, df['Prob_H=4'])
    auc_cb = average_precision_score(y_true, df['Prob_H=4'])
    
    baseline = y_true.sum() / len(y_true)

    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [baseline, baseline], label=f'Baseline Prevalence (PR-AUC {baseline:.2f})', linestyle=':', color='gray')
    plt.plot(r_lr, p_lr, label=f'Logistic Econometric (PR-AUC {auc_lr:.2f})', linestyle='-.', color='orange')
    plt.plot(r_lgbm, p_lgbm, label=f'LightGBM Challenger (PR-AUC {auc_lgbm:.2f})', linestyle='--', color='blue')
    plt.plot(r_cb, p_cb, label=f'CatBoost Primary (PR-AUC {auc_cb:.2f})', linewidth=2.5, color='darkred')

    plt.title('Precision-Recall Curves (1-Year Horizon)', fontsize=14, pad=15)
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.legend(loc='lower left', fontsize=11, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    f12_path = os.path.join(OUT_DIR, "F12_Opposition_PR.png")
    plt.savefig(f12_path, dpi=300)
    print(f"[+] Successfully saved {f12_path}")

if __name__ == '__main__':
    plot_f12()
