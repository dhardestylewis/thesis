import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.calibration import calibration_curve

# Add Scripts dir to path for thesis style
sys.path.append(os.path.abspath('Scripts'))
try:
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

ROOT_DIR = os.path.abspath('.')
REG_PATH = os.path.join(ROOT_DIR, 'registries', 'prediction_registry.parquet')
FIG_DIR = os.path.join(ROOT_DIR, 'Thesis_Draft', 'Draft_v1', 'Figures', 'exhibits')
os.makedirs(FIG_DIR, exist_ok=True)

# Try fetching new title from json
title_str = "Formal Petition Reliability (Filing Date Baseline Out-of-Fold)"
try:
    import json
    with open(os.path.join(ROOT_DIR, "Scripts", "exhibit_titles.json"), "r") as f:
        titles = json.load(f)
        if "track1_OOF_stage_c" in titles:
            title_str = titles["track1_OOF_stage_c"] + " (Filing Date Baseline Out-of-Fold)"
except Exception:
    pass

df = pd.read_parquet(REG_PATH)
df = df[df['role'] == 'test']

y_true = df['y_true'].astype(float)
y_prob = df['y_score_calibrated'].astype(float)

prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.plot(prob_pred, prob_true, 'o-', color='navy', label='CatBoost Primary (V-REx)', linewidth=2)

plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives (Empirical)')
plt.title(title_str)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_file = os.path.join(FIG_DIR, "fig_calibration_ece_H0.pdf")
plt.savefig(out_file)
print(f"Saved: {out_file}")
