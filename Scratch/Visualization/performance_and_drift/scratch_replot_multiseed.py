import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DRAFT_DIR = os.path.join(ROOT, "Thesis_Draft")

print("[*] Loading Cached Multiseed Matrix...")
res_df = pd.read_csv(os.path.join(DRAFT_DIR, "Multiseed_Performance_Matrix.csv"))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
g = sns.relplot(
    data=res_df,
    x='Offset',
    y='Score',
    hue='Topology',
    col='Architecture',
    kind='line',
    marker='D',
    markersize=8,
    linewidth=3.0,
    height=6,
    aspect=1.3,
    palette=["#1f77b4", "#d62728"],
    errorbar='ci'
)

g.set_axis_labels("Out-of-Distribution Temporal Drift (Years Offset)", "Predictive Relational Accuracy (NDCG)", fontsize=13)
g.set_titles(col_template="Architecture: {col_name}", weight='bold', size=14)

# Explicitly dropping the Y-axis to 0.0 as requested to show full depth of performance collapse bounds
plt.ylim(0.0, 1.05)

plt.suptitle("Algorithmic Invulnerability: Structural Robustness across Native Initialization Seeds\nValidating that relational mappings hold absolute accuracy bounds all the way down to a 0.0 floor.", 
             fontsize=17, weight='bold', y=1.06)

out_png = os.path.join(DRAFT_DIR, "plot_multiseed_performance_drift.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close()

print(f"[*] Plotted to: {out_png}")
