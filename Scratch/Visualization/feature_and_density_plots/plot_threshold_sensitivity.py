import os
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

root = r'C:\Users\dhl\data\Thesis\thesis'
drift_dir = os.path.join(root, 'Thesis_Draft', 'Draft_v1', 'Tables', 'appendices_drift')

thresholds = [0, 5, 10, 15, 20, 25]
data = []

anchors = ['Pre-2018', 'Pre-2019', 'Pre-2020', 'Pre-2021', 'Pre-2022', 'Pre-2023', 'Pre-2024']
years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']

for t in thresholds:
    file_path = os.path.join(drift_dir, f'tbl_ch4_15_temporal_drift_family_t{t:02d}.tex')
    if not os.path.exists(file_path):
        continue
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    in_panel_a = False
    for line in lines:
        if 'Panel A: Maximum Absolute PR-AUC' in line:
            in_panel_a = True
            continue
        if in_panel_a and 'Panel B' in line:
            break
        if in_panel_a and line.startswith('Pre-'):
            parts = line.strip().split('&')
            anchor = parts[0].strip()
            for i, test_year in enumerate(years):
                if i + 1 < len(parts):
                    cell = parts[i+1].strip()
                    if cell.startswith('---'):
                        continue
                    # Extract the numerical value, e.g., "0.994 (Causal)" or "\textbf{0.794} (Trees)"
                    m = re.search(r'([0-9]\.[0-9]{3})', cell)
                    if m:
                        val = float(m.group(1))
                        data.append({'Threshold': t, 'Anchor': anchor, 'Test_Year': test_year, 'PRAUC': val})

import pandas as pd
df = pd.DataFrame(data)

# Create a faceted grid plot by Anchor Year
sns.set_theme(style='whitegrid')
g = sns.FacetGrid(df, col='Anchor', col_wrap=4, sharex=False, height=3.5, aspect=1.2)

def plot_lines(x, y, color, label, **kwargs):
    sns.lineplot(x=x, y=y, marker='o', **kwargs)

g.map_dataframe(sns.lineplot, x='Threshold', y='PRAUC', hue='Test_Year', marker='o', palette='tab10')
g.set_axis_labels('Petition Threshold (%)', 'PR-AUC')
g.add_legend(title='Test Year')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Sensitivity of Dominant PR-AUC by Threshold (Max-of-Family)', fontsize=16)

out_file = os.path.join(root, 'Scratch', 'threshold_sensitivity_plot.png')
g.savefig(out_file, dpi=300, bbox_inches='tight')
print(f'Saved plot to {out_file}')
