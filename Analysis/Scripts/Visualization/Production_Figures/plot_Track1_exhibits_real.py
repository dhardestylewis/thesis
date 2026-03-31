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

from sklearn.calibration import calibration_curve
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
STAGE_C_OUT = os.path.join(ROOT, "Analysis", "Output", "Track1_Predictive")
FIG_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Track1_Exhibits")
os.makedirs(FIG_DIR, exist_ok=True)

def plot_all_track1_exhibits():
    for hz in ['H0', 'H3']:
        print("==============================================")
        print(f" Rendering Authentic Track 1 PDF Exhibits: {hz}")
        print("==============================================")

        # 1. Reliability Diagram (Calibration & ECE)
        preds_file = os.path.join(STAGE_C_OUT, f"stage_c_oof_predictions_{hz}.csv")
        if os.path.exists(preds_file):
            df_oof = pd.read_csv(preds_file)
            prob_true, prob_pred = calibration_curve(df_oof['y_true'], df_oof['y_prob'], n_bins=10)
            
            plt.figure(figsize=(7, 6))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            plt.plot(prob_pred, prob_true, 's-', color='darkred', label=f'CatBoost ({hz})')
            plt.title(f'Stage C Opposition Reliability ({hz} Out-of-Fold)', fontsize=14)
            plt.xlabel('Mean Predicted Probability', fontsize=12)
            plt.ylabel('Fraction of Positives', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"fig_calibration_ece_{hz}.pdf"))
            print(f"  [+] Saved fig_calibration_ece_{hz}.pdf")
            
        # 2. Temporal Drift
        drift_file = os.path.join(STAGE_C_OUT, f"stage_c_drift_{hz}.csv")
        if os.path.exists(drift_file):
            plt.figure(figsize=(7, 5))
            try:
                df_drift = pd.read_csv(drift_file)
                if not df_drift.empty:
                    for anchor in df_drift['Anchor'].unique():
                        sub = df_drift[df_drift['Anchor'] == anchor]
                        plt.plot(sub['Offset'], sub['PR-AUC'], marker='o', label=f'Anchor < {anchor}')
                    plt.title(f'Temporal Predictive Drift ({hz} Rolling Origin)', fontsize=14)
                    plt.xlabel('Years Out-of-Distribution (T + offset)', fontsize=12)
                    plt.ylabel('PR-AUC', fontsize=12)
                    plt.xticks([0, 1, 2, 3])
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            except:
                pass
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"fig_temporal_drift_{hz}.pdf"))
            print(f"  [+] Saved fig_temporal_drift_{hz}.pdf")
            
        # 3. Policy Regimes
        regimes_file = os.path.join(STAGE_C_OUT, f"stage_c_regimes_{hz}.csv")
        if os.path.exists(regimes_file):
            plt.figure(figsize=(8, 5))
            try:
                df_reg = pd.read_csv(regimes_file)
                if not df_reg.empty:
                    plt.bar(df_reg['Regime'], df_reg['PR-AUC'], color=['navy', 'orange', 'darkred'])
                    plt.title(f'Out-of-Distribution Policy Regime Degradation ({hz})', fontsize=14)
                    plt.ylabel('PR-AUC', fontsize=12)
                    plt.ylim(0, max(0.5, df_reg['PR-AUC'].max() * 1.2))
                    plt.grid(axis='y', alpha=0.3)
            except:
                pass
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"fig_policy_regimes_{hz}.pdf"))
            print(f"  [+] Saved fig_policy_regimes_{hz}.pdf")

        # 4. Feature Importance
        fi_file = os.path.join(STAGE_C_OUT, f"stage_c_feature_importance_{hz}.csv")
        if os.path.exists(fi_file):
            try:
                df_fi = pd.read_csv(fi_file).head(15)
                # Sort ascending for horizontal bar chart
                df_fi = df_fi.sort_values('Importance', ascending=True)
                plt.figure(figsize=(10, 8))
                plt.barh(df_fi['Feature'], df_fi['Importance'], color='darkblue', alpha=0.8)
                plt.title(f'Top 15 Native Feature Importances ({hz})', fontsize=14)
                plt.xlabel('Relative Importance (%)', fontsize=12)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, f"fig_feature_importance_{hz}.pdf"))
                print(f"  [+] Saved fig_feature_importance_{hz}.pdf")
            except:
                pass

if __name__ == '__main__':
    plot_all_track1_exhibits()
