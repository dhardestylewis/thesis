import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import os

df = pd.read_csv(r'c:\Users\dhl\data\thesis\thesis\Analysis\Output\Track1_Predictive\stage_c_oof_predictions.csv')
y_true = df['y_true']
y_prob = df['y_prob']
dists = df['district']

prauc = average_precision_score(y_true, y_prob)

prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
ece = np.mean(np.abs(prob_true - prob_pred))
z = np.polyfit(prob_pred, prob_true, 1)
calib_slope = z[0]

# Lift
df_eval = pd.DataFrame({'y': y_true, 'p': y_prob})
base_rate = df_eval['y'].mean()
df_eval = df_eval.sort_values('p', ascending=False)
k = max(1, int(len(df_eval) * 0.10))
top_decile_hit_rate = df_eval.head(k)['y'].mean()
lift = top_decile_hit_rate / base_rate

# FNR Gap
threshold = np.percentile(y_prob, 50)
df_eval['pred_binary'] = (y_prob > threshold).astype(int)
df_eval['d'] = dists
fnrs = {}
for d in df_eval['d'].unique():
    sub = df_eval[df_eval['d'] == d]
    positives = sub[sub['y'] == 1]
    if len(positives) > 0:
        fnrs[d] = 1.0 - (positives['pred_binary'].sum() / len(positives))
fnr_gap = (max(fnrs.values()) - min(fnrs.values())) * 100

print(f"Total N = {len(df)}")
print(f"PR-AUC: {prauc:.3f}")
print(f"ECE: {ece:.3f}")
print(f"Calib-Slope: {calib_slope:.3f}")
print(f"Top-Decile Lift: {lift:.3f}")
print(f"FNR Gap: {fnr_gap:.2f}%")
