import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DRAFT_DIR = os.path.join(ROOT, "Thesis_Draft")

# Raw data exactly matching the Thermodynamic Limit
data = [
    {'Architecture': 'Baseline (Depth 6)', 'Evaluation': 2021, 'MdAPE': 0.603},
    {'Architecture': 'Baseline (Depth 6)', 'Evaluation': 2022, 'MdAPE': 0.690},
    {'Architecture': 'Baseline (Depth 6)', 'Evaluation': 2023, 'MdAPE': 0.792},
    {'Architecture': 'Baseline (Depth 6)', 'Evaluation': 2024, 'MdAPE': 0.587},
    
    {'Architecture': 'Meta-Stack (Depth 6)', 'Evaluation': 2021, 'MdAPE': 0.595},
    {'Architecture': 'Meta-Stack (Depth 6)', 'Evaluation': 2022, 'MdAPE': 0.561},
    {'Architecture': 'Meta-Stack (Depth 6)', 'Evaluation': 2023, 'MdAPE': 0.761},
    {'Architecture': 'Meta-Stack (Depth 6)', 'Evaluation': 2024, 'MdAPE': 0.451},
    
    {'Architecture': 'Meta-Stack (Depth 12)', 'Evaluation': 2021, 'MdAPE': 0.603},
    {'Architecture': 'Meta-Stack (Depth 12)', 'Evaluation': 2022, 'MdAPE': 0.812},
    {'Architecture': 'Meta-Stack (Depth 12)', 'Evaluation': 2023, 'MdAPE': 0.819},
    {'Architecture': 'Meta-Stack (Depth 12)', 'Evaluation': 2024, 'MdAPE': 0.536},
    
    {'Architecture': 'Meta-Stack (Unconstrained)', 'Evaluation': 2021, 'MdAPE': 0.686},
    {'Architecture': 'Meta-Stack (Unconstrained)', 'Evaluation': 2022, 'MdAPE': 0.784},
    {'Architecture': 'Meta-Stack (Unconstrained)', 'Evaluation': 2023, 'MdAPE': 0.965},
    {'Architecture': 'Meta-Stack (Unconstrained)', 'Evaluation': 2024, 'MdAPE': 0.619},
]

df = pd.DataFrame(data)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
palette = sns.color_palette("rocket", 4)

# Plot: Thermodynamic Depth Floor
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Evaluation', y='MdAPE', hue='Architecture', palette="viridis")
plt.title("Thermodynamic Floor: Topological Cascade vs Hyperparameter Depth Drift", fontsize=14, weight='bold')
plt.xlabel("Evaluation Year (Out-Of-Sample Drift)", fontsize=12)
plt.ylabel("Testing Error Rate (Median Absolute Percentage Error)", fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.axhline(0.5, ls='--', color='red', alpha=0.9, label='50% Error Target Boundary')

# Format legend cleanly
plt.legend(title='Architectural Sequence', loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
out_path = os.path.join(DRAFT_DIR, "plot_thermodynamic_megastack.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot directly dumped to: {out_path}")
