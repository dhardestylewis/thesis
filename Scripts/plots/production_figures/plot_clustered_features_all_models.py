import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

ROOT = r"C:\Users\dhl\data\thesis\thesis"
OUT_PDF = os.path.join(ROOT, "Thesis_Draft", "Draft_v1", "Figures", "exhibits", "fig_feature_importance_clustered_H0_Full.pdf")
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

# Data sourced directly from tbl_ch5_02_archetypal_attribution.tex
data = {
    'Semantic Target Cluster': [
        'Parcel Scale', 'Zoning Density', 'Structure Age', 'Neighborhood Income', 
        'Demographics', 'Improvement Scale', 'Housing Tenure', 'Neighborhood Valuation', 'Property Valuation'
    ],
    'Tree Ensembles': [44.4, 36.1, 7.5, 3.7, 3.0, 2.4, 1.5, 1.3, 0.1],
    'Deep Architectures': [54.7, 5.3, 2.9, 1.2, 17.6, 1.8, 11.7, 1.1, 3.7],
    'Linear Architectures': [23.8, 20.3, 9.1, 7.2, 15.0, 10.1, 6.5, 1.2, 6.7],
    'Causal Algorithms': [65.4, 5.5, 3.8, 3.7, 7.9, 3.0, 3.7, 6.4, 0.5]
}
df = pd.DataFrame(data)

# Melt for seaborn
df_melt = df.melt(id_vars='Semantic Target Cluster', var_name='Architecture', value_name='Reliance')

g = sns.FacetGrid(df_melt, col="Architecture", col_wrap=2, height=4.5, aspect=1.3, sharex=True)

def plot_bar(x, y, color, **kwargs):
    ax = plt.gca()
    # Sort data for this specific architecture
    data = kwargs.pop('data')
    data = data.sort_values(by=x, ascending=True)
    sns.barplot(data=data, x=x, y=y, color=color, alpha=0.85, ax=ax, edgecolor='black', linewidth=0.5)

colors = {'Tree Ensembles': 'darkred', 'Deep Architectures': 'dodgerblue', 'Linear Architectures': 'forestgreen', 'Causal Algorithms': 'purple'}

g.map_dataframe(plot_bar, x='Reliance', y='Semantic Target Cluster', color='dodgerblue')

# Apply individual colors
for i, ax in enumerate(g.axes):
    arch = df_melt['Architecture'].unique()[i]
    for bar in ax.patches:
        bar.set_facecolor(colors[arch])
    ax.set_title(f"{arch}", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Relative Attribution Share (%)", fontsize=11)
    ax.set_ylabel("")
    ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.suptitle("Comparative Primary Reliance: Top Feature Clusters Across Architectures", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight')
print(f"[+] Saved: {OUT_PDF}")
