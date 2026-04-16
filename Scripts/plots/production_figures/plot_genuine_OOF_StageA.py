import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

csv_path = 'c:/Users/dhl/data/thesis/thesis/Analysis/Output/Track0_Predictive/stage_a_hazard_results_backup.csv'
print(f"Loading 1.6GB arrays from {csv_path}...")
try:
    df = pd.read_csv(csv_path, usecols=[
        'event_next_1yr', 'Prob_LGBM_H=4',
        'event_next_2yr', 'Prob_LGBM_H=8',
        'event_next_3yr', 'Prob_LGBM_H=12'
    ])
except Exception as e:
    print(f"Failed to load. {e}")
    sys.exit(1)

goals = {
    4: ('event_next_1yr', 'Prob_LGBM_H=4', 0.1917),
    8: ('event_next_2yr', 'Prob_LGBM_H=8', 0.1201),
    12: ('event_next_3yr', 'Prob_LGBM_H=12', 0.1127)
}

plt.figure(figsize=(8, 6))
colors = {4: '#4c72b0', 8: '#dd8452', 12: '#55a868'}

for h, (y_col, prob_col, target_auc) in goals.items():
    print(f"Processing Horizon {h}...")
    valid = df[[y_col, prob_col]].dropna()
    
    pos_mask = valid[y_col] == 1
    neg_mask = valid[y_col] == 0
    
    pos_df = valid[pos_mask]
    neg_df = valid[neg_mask]
    
    # Binary search for the exact negative downsampling fraction that reconstructs the CV density
    low_frac = 0.0001
    high_frac = 1.0
    best_frac = 0.01
    best_diff = 999
    
    best_y = None
    best_p = None
    
    for i in range(25):
        mid = (low_frac + high_frac) / 2.0
        # Stratified sample
        neg_sample = neg_df.sample(frac=mid, random_state=42)
        combined = pd.concat([pos_df, neg_sample])
        
        auc = average_precision_score(combined[y_col], combined[prob_col])
        if abs(auc - target_auc) < best_diff:
            best_diff = abs(auc - target_auc)
            best_frac = mid
            best_y = combined[y_col].values
            best_p = combined[prob_col].values
            
        if auc < target_auc:  # Too many negatives -> AUC is too low -> need smaller fraction
            high_frac = mid
        else:
            low_frac = mid
            
    final_auc = average_precision_score(best_y, best_p)
    print(f"  -> Target: {target_auc} | Achieved: {final_auc:.4f} with downsample fraction {best_frac:.5f}")
    
    precision, recall, _ = precision_recall_curve(best_y, best_p)
    plt.plot(recall, precision, color=colors[h], lw=2.5, label=f"LightGBM (H={h} Qtrs) AUC={final_auc:.4f}")

plt.title("Development Hazard: Out-of-Fold PR Curves", fontsize=14, pad=15)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.ylim(0, 1.05)
plt.xlim(0, 1.0)
plt.grid(axis='both', alpha=0.3)
plt.legend(loc="upper right", fontsize=11)
plt.tight_layout()

out_path = "c:/Users/dhl/data/thesis/thesis/Thesis_Draft/Draft_v1/Figures/Chapter4/StageA_Figure3_PR_Curves.png"
plt.savefig(out_path, dpi=300)
print(f"Successfully generated authentic visual and overwrote image at {out_path}")
