import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
res_df = pd.read_csv(os.path.join(ROOT, 'grid_tuning_results.csv'))

# Compare average PR-AUC of the Pre-2020 models across all evaluation years.
mean_pre2020 = res_df.groupby(['Model', 'Profile'])['PRAUC'].mean().round(3).sort_values()

print("AVERAGE OOS PR-AUC ACROSS ALL FUTURE YEARS (PRE-2020 ANCHOR):")
print(mean_pre2020)

