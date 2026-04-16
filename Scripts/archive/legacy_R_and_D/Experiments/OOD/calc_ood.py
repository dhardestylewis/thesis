import pandas as pd
import numpy as np
import os

df = pd.read_csv(r'c:\Users\dhl\data\thesis\thesis\Analysis\Results\modal_robustness_results.csv')
paired = df.pivot(index=['seed', 'arch_id', 'latent_dim'], columns='method', values='pr_auc').reset_index()
paired['diff'] = paired['V-REx'] - paired['ERM']

print('=' * 80)
print('MODAL ROBUSTNESS OOD RESULTS (N = 15 Architecture/Seed Variations)')
print('=' * 80)
for _, row in paired.iterrows():
    print(f"Seed {int(row['seed']):4d} | Arch: {int(row['latent_dim']):2d}-dim | ERM PR: {row['ERM']:.4f} | V-REx PR: {row['V-REx']:.4f} | Diff: {row['diff']:+.4f}")
print('-' * 80)
erm_mean = paired['ERM'].mean()
erm_std = paired['ERM'].std()
vrex_mean = paired['V-REx'].mean()
vrex_std = paired['V-REx'].std()
diff_mean = paired['diff'].mean()
diff_std = paired['diff'].std()

t_stat = diff_mean / (diff_std / np.sqrt(len(paired)))

print(f"ERM   Mean PR-AUC: {erm_mean:.4f} +/- {erm_std:.4f}")
print(f"V-REx Mean PR-AUC: {vrex_mean:.4f} +/- {vrex_std:.4f}")
print(f"Average Improvement (VREx - ERM): {diff_mean:+.4f}")
print(f"t-statistic: {t_stat:.4f}")

if diff_mean > 0 and t_stat > 1.76:
    print("RESULT: STATISTICALLY SIGNIFICANT IMPROVEMENT (p < 0.05)")
elif diff_mean > 0:
    print("RESULT: IMPROVEMENT NOT STATISTICALLY SIGNIFICANT")
else:
    print("RESULT: V-REx FAILED TO GENERALIZE BETTER THAN ERM")
