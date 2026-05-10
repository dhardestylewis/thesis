import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
res_df = pd.read_csv(os.path.join(ROOT, 'grid_tuning_results_full.csv'))

# Mean PR-AUC across ALL Out-Of-Sample drift periods across ALL anchors
universal_mean = res_df.groupby(['Model', 'Profile'])['PRAUC'].mean().round(3).reset_index()
universal_mean = universal_mean.sort_values(by='PRAUC', ascending=False)
markdown_table = universal_mean.to_markdown(index=False)

with open(os.path.join(ROOT, 'grid_results_universal_markdown.txt'), 'w') as f:
    f.write(markdown_table)

print(markdown_table)
