import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
df_expanded = pd.read_csv(os.path.join(ROOT, 'grid_tuning_results_expanded.csv'))

# Isolate Pre-2020 anchor for fair comparison
pre2020 = df_expanded[df_expanded['Anchor'] == 'Pre-2020']

# Get the average PRAUC across the 4 forward years (2021-2024) for each Model+Profile
mean_perf = pre2020.groupby(['Model', 'Profile'])['PRAUC'].mean().round(3).reset_index()

# Filter just CatBoost, LightGBM, RandomForest
trees = mean_perf[mean_perf['Model'].isin(['CatBoost', 'LightGBM', 'RandomForest'])].sort_values(by=['Model', 'Profile'])

print("--- PRE-2020 ANCHOR AVERAGE OOS PR-AUC (2021-2024) ---")
print(trees.to_markdown(index=False))

