import pandas as pd
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
res_df = pd.read_csv(os.path.join(ROOT, 'grid_tuning_results.csv'))

# Create a pivot table to show the exact Trajectories
pivot = res_df.pivot_table(index=['Model', 'Profile'], columns='Evaluate_Year', values='PRAUC')
pivot = pivot.round(3)
markdown_table = pivot.to_markdown()

with open(os.path.join(ROOT, 'grid_results_markdown.txt'), 'w') as f:
    f.write(markdown_table)

print(markdown_table)
