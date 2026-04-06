import pandas as pd
import numpy as np
import re

OOF_PATH = r"Analysis\Output\Track1_Predictive\Metrics\stage_c_oof_predictions_H0.csv"
METRICS_TEX = r"Thesis_Draft\Draft_v1\Tables\metrics_config.tex"

df = pd.read_csv(OOF_PATH)
y_true = df['y_true'].values
y_prob = df['y_prob'].values

def compute_ace(y_true, y_prob, n_bins=10):
    """Adaptive Calibration Error using equal-mass bins."""
    sorted_idx = np.argsort(y_prob)
    y_prob_sorted = y_prob[sorted_idx]
    y_true_sorted = y_true[sorted_idx]
    
    bin_size = len(y_prob) // n_bins
    ace = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_prob)
        
        bin_prob = y_prob_sorted[start:end]
        bin_true = y_true_sorted[start:end]
        
        if len(bin_prob) > 0:
            ace += (len(bin_prob) / len(y_prob)) * abs(bin_prob.mean() - bin_true.mean())
            
    return ace

ace_val = compute_ace(y_true, y_prob)

# Bootstrap ACE
np.random.seed(42)
n_boot = 2000
ace_boots = []
for _ in range(n_boot):
    idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
    ace_boots.append(compute_ace(y_true[idx], y_prob[idx]))

ace_lo = np.percentile(ace_boots, 2.5)
ace_hi = np.percentile(ace_boots, 97.5)

print(f"ACE: {ace_val:.4f} 95% CI: [{ace_lo:.3f}, {ace_hi:.3f}]")

macro_string = f"\n% ACE Metric computed via equal-mass binning\n\\newcommand{{\\metricACE}}{{{ace_val:.4f}}}\n\\newcommand{{\\metricACEBootCI}}{{[{ace_lo:.3f}, {ace_hi:.3f}]}}\n"

with open(METRICS_TEX, 'a') as f:
    f.write(macro_string)
