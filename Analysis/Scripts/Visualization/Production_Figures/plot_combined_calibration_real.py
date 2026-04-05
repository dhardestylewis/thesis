import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os

try:
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

from artifact_registry import ROOT_DIR, TraceabilityRegistry as AR
ROOT = str(ROOT_DIR)
OUT_DIR = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "Chapter4")
os.makedirs(OUT_DIR, exist_ok=True)
STAGE_A_OUT = str(AR.STAGE_A_HAZARD_RESULTS)
STAGE_C_OUT = str(AR.stage_c_oof("H0"))


def plot_combined_calibration():
    print("==============================================")
    print(" Rendering Combined Calibration Grid")
    print("==============================================")

    if not os.path.exists(STAGE_A_OUT) or not os.path.exists(STAGE_C_OUT):
        print("[-] Required predictive data not found.")
        return

    # ── Load Stage A ──────────────────────────────────────────────────
    df_a = pd.read_csv(STAGE_A_OUT,
                       usecols=['event_next_1yr', 'Prob_LR_H=4', 'Prob_Optimal_H=4'])
    y_true_a = df_a['event_next_1yr']

    try:
        with open(str(AR.stage_a_winner('H=4'))) as f:
            optimal_name_a = f.read().strip()
    except Exception:
        optimal_name_a = "Optimal Champion"

    prob_true_opt, prob_pred_opt = calibration_curve(
        y_true_a, df_a['Prob_Optimal_H=4'], n_bins=10)
    prob_true_lr_a, prob_pred_lr_a = calibration_curve(
        y_true_a, df_a['Prob_LR_H=4'], n_bins=10)

    # ── Load Stage C ──────────────────────────────────────────────────
    df_c = pd.read_csv(STAGE_C_OUT,
                       usecols=['y_true', 'y_prob', 'y_prob_lr', 'y_prob_rf',
                                'y_prob_spatial_lr', 'y_prob_anchor'])
    y_true_c = df_c['y_true']

    # Calibration curves for each Stage C model
    cal_c = {}
    for col, label in [('y_prob_lr',         'Standard Logistic (ERM)'),
                       ('y_prob_rf',         'RandomForest (ERM)'),
                       ('y_prob_spatial_lr', 'Spatial-FE Logistic (Domain)'),
                       ('y_prob_anchor',     'Anchor Regression (Causal)'),
                       ('y_prob',            'CatBoost Primary (V-REx)')]:
        pt, pp = calibration_curve(y_true_c, df_c[col], n_bins=10)
        cal_c[label] = (pt, pp)

    # ── Consistent colour / linestyle (matches F12 PR curves) ─────────
    STYLE_C = {
        'Standard Logistic (ERM)':        dict(color='coral',   linestyle=':', marker='v', lw=1.2),
        'RandomForest (ERM)':             dict(color='gray',    linestyle=':', marker='^', lw=1.2),
        'Spatial-FE Logistic (Domain)':   dict(color='purple',  linestyle='--', marker='D', lw=1.5),
        'Anchor Regression (Causal)':     dict(color='teal',    linestyle='-.', marker='x', lw=1.5),
        'CatBoost Primary (V-REx)':       dict(color='darkred', linestyle='-',  marker='s', lw=2.5),
    }

    # ── Build 1×3 grid ────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel (a): Stage A Reliability Diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.plot(prob_pred_opt, prob_true_opt, marker='o', lw=2,
             color='darkblue', label=f'{optimal_name_a} (V-REx)')
    ax1.plot(prob_pred_lr_a, prob_true_lr_a, marker='s', linestyle=':',
             color='gray', label='Logistic Baseline')
    ax1.set_title("(a) Stage A: Calibration Reliability")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Stage A Capture Curve (Gains)
    df_sort_opt = df_a.sort_values('Prob_Optimal_H=4', ascending=False).reset_index(drop=True)
    df_sort_opt['cum'] = df_sort_opt['event_next_1yr'].cumsum()
    df_sort_lr = df_a.sort_values('Prob_LR_H=4', ascending=False).reset_index(drop=True)
    df_sort_lr['cum'] = df_sort_lr['event_next_1yr'].cumsum()

    total_ev = y_true_a.sum()
    pcts = np.linspace(0, 100, len(df_a))

    ax2.plot(pcts, (df_sort_opt['cum'] / total_ev) * 100,
             lw=2, color='darkblue', label=f'{optimal_name_a} Capture')
    ax2.plot(pcts, (df_sort_lr['cum'] / total_ev) * 100,
             linestyle='--', color='gray', label='Logistic Capture')
    ax2.plot([0, 100], [0, 100], linestyle=':', color='black', label='Random Baseline')
    ax2.set_title("(b) Stage A: Capture Curve (Gains Rate)")
    ax2.set_xlabel("Top Percentile of Ranked Sites")
    ax2.set_ylabel("% of Realized Events Captured")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel (c): Stage C Reliability Diagram – full 5-model gauntlet
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    for label, (pt, pp) in cal_c.items():
        s = STYLE_C[label]
        ax3.plot(pp, pt, label=label, **s)
    ax3.set_title("(c) Stage C: Opposition Reliability (H0)")
    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("Fraction of Positives")
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_pdf = os.path.join(OUT_DIR, "fig_combined_calibration_reliability.pdf")
    out_png = os.path.join(OUT_DIR, "fig_combined_calibration_reliability.png")
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[+] Saved {out_pdf}")
    print(f"[+] Saved {out_png}")


if __name__ == '__main__':
    plot_combined_calibration()
