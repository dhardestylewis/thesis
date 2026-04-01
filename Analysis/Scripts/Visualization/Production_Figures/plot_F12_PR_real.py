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

import sys
_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from artifact_registry import ROOT_DIR, FIGURES_DIR, TraceabilityRegistry as AR

ROOT = str(ROOT_DIR)
OUT_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
os.makedirs(OUT_DIR, exist_ok=True)

# Stage C Opposition Results contain the authentic out-of-fold probabilistic outputs
STAGE_C_OUT = str(AR.STAGE_C_OOF_H0)

def plot_f12():
    print("==============================================")
    print(" Rendering Authentic F12: PR Curves")
    print("==============================================")
    
    if not os.path.exists(STAGE_C_OUT):
        print("[-] Required Stage C predictive data not found.")
        return
        
    df = pd.read_csv(STAGE_C_OUT, usecols=['y_true', 'y_prob_lr', 'y_prob_rf', 'y_prob_spatial_lr', 'y_prob_anchor', 'y_prob'])
    
    # We evaluate for the H=4 (1 Yr) horizon
    y_true = df['y_true']
    
    # Calculate PR Curves for the underlying empirical models
    p_lr, r_lr, _ = precision_recall_curve(y_true, df['y_prob_lr'])
    auc_lr = average_precision_score(y_true, df['y_prob_lr'])
    
    p_rf, r_rf, _ = precision_recall_curve(y_true, df['y_prob_rf'])
    auc_rf = average_precision_score(y_true, df['y_prob_rf'])
    
    p_sp, r_sp, _ = precision_recall_curve(y_true, df['y_prob_spatial_lr'])
    auc_sp = average_precision_score(y_true, df['y_prob_spatial_lr'])
    
    p_anc, r_anc, _ = precision_recall_curve(y_true, df['y_prob_anchor'])
    auc_anc = average_precision_score(y_true, df['y_prob_anchor'])
    
    p_cb, r_cb, _ = precision_recall_curve(y_true, df['y_prob'])
    auc_cb = average_precision_score(y_true, df['y_prob'])
    
    baseline = y_true.sum() / len(y_true)

    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [baseline, baseline], label=f'Baseline Prevalence (PR-AUC {baseline:.2f})', linestyle=':', color='gray')
    plt.plot(r_lr, p_lr, label=f'Standard Logistic (ERM) (PR-AUC {auc_lr:.2f})', linestyle=':', color='coral')
    plt.plot(r_rf, p_rf, label=f'RandomForest (ERM) (PR-AUC {auc_rf:.2f})', linestyle=':', color='gray')
    plt.plot(r_sp, p_sp, label=f'Spatial-FE Logistic (Domain) (PR-AUC {auc_sp:.2f})', linestyle='--', color='purple')
    plt.plot(r_anc, p_anc, label=f'Anchor Regression (Causal) (PR-AUC {auc_anc:.2f})', linestyle='-.', color='teal', linewidth=1.5)
    plt.plot(r_cb, p_cb, label=f'CatBoost Primary (V-REx) (PR-AUC {auc_cb:.2f})', linewidth=2.5, color='darkred')

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
