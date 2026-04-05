"""
Supplemental metrics computation for thesis Point 7 gaps.
Computes: Brier score, exact top-decile precision, recall at threshold,
subgroup Ns per council district, and bootstrap CI on ECE.
Writes new macros to metrics_config.tex.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, precision_score, recall_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# ---- Load OOF predictions ----
OOF_PATH = r"Analysis\Output\Track1_Predictive\Metrics\stage_c_oof_predictions_H0.csv"
METRICS_TEX = r"Thesis_Draft\Draft_v1\Tables\metrics_config.tex"

df = pd.read_csv(OOF_PATH)
y_true = df['y_true'].values
y_prob = df['y_prob'].values  # CatBoost post-isotonic probabilities

print(f"Loaded {len(df)} cases, {y_true.sum():.0f} petitions, base rate = {y_true.mean():.4f}")

# ---- 1. Brier Score ----
brier = brier_score_loss(y_true, y_prob)
print(f"\nBrier Score: {brier:.4f}")

# ---- 2. Exact Top-Decile Precision ----
n_decile = len(df) // 10
top_decile_idx = np.argsort(y_prob)[-n_decile:]
top_decile_precision = y_true[top_decile_idx].mean()
print(f"Top-decile precision: {top_decile_precision:.3f} ({y_true[top_decile_idx].sum():.0f}/{n_decile})")

# ---- 3. Recall at P >= 0.5 threshold ----
y_pred_50 = (y_prob >= 0.5).astype(int)
if y_pred_50.sum() > 0:
    precision_50 = precision_score(y_true, y_pred_50, zero_division=0)
    recall_50 = recall_score(y_true, y_pred_50, zero_division=0)
    n_flagged_50 = y_pred_50.sum()
else:
    precision_50 = 0.0
    recall_50 = 0.0
    n_flagged_50 = 0
print(f"At P >= 0.50 threshold: precision={precision_50:.3f}, recall={recall_50:.3f}, N flagged={n_flagged_50}")

# Also try 0.3 as a lower actionable threshold
y_pred_30 = (y_prob >= 0.3).astype(int)
if y_pred_30.sum() > 0:
    precision_30 = precision_score(y_true, y_pred_30, zero_division=0)
    recall_30 = recall_score(y_true, y_pred_30, zero_division=0)
    n_flagged_30 = y_pred_30.sum()
else:
    precision_30 = 0.0
    recall_30 = 0.0
    n_flagged_30 = 0
print(f"At P >= 0.30 threshold: precision={precision_30:.3f}, recall={recall_30:.3f}, N flagged={n_flagged_30}")

# ---- 4. Subgroup Ns per Council District ----
print("\nSubgroup breakdown by council district:")
subgroup = df.groupby('district').agg(
    N=('y_true', 'count'),
    petitions=('y_true', 'sum'),
    base_rate=('y_true', 'mean')
).reset_index()
subgroup['base_rate'] = subgroup['base_rate'].map(lambda x: f"{x:.3f}")
print(subgroup.to_string(index=False))

# Count districts with >= 10 positive cases (meaningful for FNR)
n_districts_meaningful = (subgroup['petitions'] >= 10).sum()
n_districts_total = len(subgroup)
min_positives = int(subgroup['petitions'].min())
max_positives = int(subgroup['petitions'].max())
median_positives = int(subgroup['petitions'].median())
print(f"\nDistricts with >= 10 positives: {n_districts_meaningful}/{n_districts_total}")
print(f"Positives per district: min={min_positives}, median={median_positives}, max={max_positives}")

# ---- 5. Bootstrap CI on ECE ----
def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece

ece_point = compute_ece(y_true, y_prob)
print(f"\nECE (point estimate): {ece_point:.4f}")

# Bootstrap
np.random.seed(42)
n_boot = 2000
ece_boots = []
for _ in range(n_boot):
    idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
    ece_boots.append(compute_ece(y_true[idx], y_prob[idx]))

ece_boots = np.array(ece_boots)
ece_lo = np.percentile(ece_boots, 2.5)
ece_hi = np.percentile(ece_boots, 97.5)
print(f"ECE 95% Bootstrap CI: [{ece_lo:.3f}, {ece_hi:.3f}]")

# ---- 6. Write new macros ----
new_macros = f"""
% Supplemental Point-7 Metrics (computed by supplemental_metrics.py)
\\newcommand{{\\metricBrierScore}}{{{brier:.4f}}}
\\newcommand{{\\metricTopDecilePrecision}}{{{top_decile_precision:.1%}}}
\\newcommand{{\\metricTopDecileTP}}{{{int(y_true[top_decile_idx].sum())}}}
\\newcommand{{\\metricTopDecileN}}{{{n_decile}}}
\\newcommand{{\\metricRecallAtFifty}}{{{recall_50:.1%}}}
\\newcommand{{\\metricPrecisionAtFifty}}{{{precision_50:.1%}}}
\\newcommand{{\\metricNFlaggedAtFifty}}{{{int(n_flagged_50)}}}
\\newcommand{{\\metricRecallAtThirty}}{{{recall_30:.1%}}}
\\newcommand{{\\metricPrecisionAtThirty}}{{{precision_30:.1%}}}
\\newcommand{{\\metricNFlaggedAtThirty}}{{{int(n_flagged_30)}}}
\\newcommand{{\\metricBaseRate}}{{{y_true.mean():.1%}}}
\\newcommand{{\\metricBaseRateN}}{{{int(y_true.sum())}/{len(y_true)}}}
\\newcommand{{\\metricNDistricts}}{{{n_districts_total}}}
\\newcommand{{\\metricMinDistrictPositives}}{{{min_positives}}}
\\newcommand{{\\metricMedianDistrictPositives}}{{{median_positives}}}
\\newcommand{{\\metricMaxDistrictPositives}}{{{max_positives}}}
\\newcommand{{\\metricECEBootCI}}{{[{ece_lo:.3f}, {ece_hi:.3f}]}}
"""

# Append to metrics_config.tex
with open(METRICS_TEX, 'r') as f:
    existing = f.read()

# Check if we already appended
if '\\metricBrierScore' not in existing:
    with open(METRICS_TEX, 'a') as f:
        f.write(new_macros)
    print(f"\n✓ Appended {len(new_macros.strip().splitlines())} new macros to {METRICS_TEX}")
else:
    print("\n⚠ Macros already exist in metrics_config.tex, skipping append")

print("\nDone.")
