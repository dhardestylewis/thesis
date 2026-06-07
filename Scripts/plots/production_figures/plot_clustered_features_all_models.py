import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = r"C:\Users\dhl\data\thesis\thesis"
TABLE_PATH = os.path.join(ROOT, "Thesis_Draft", "GSAPP_Final_Submission", "Tables",
                          "chapter5_attribution", "tbl_ch5_02_archetypal_attribution.tex")
OUT_PDF = os.path.join(ROOT, "Thesis_Draft", "GSAPP_Final_Submission", "Figures",
                       "exhibits", "fig_feature_importance_clustered_H0_Full.pdf")
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

# Parse the live attribution table
records = []
with open(TABLE_PATH, 'r') as f:
    for line in f:
        line = line.strip()
        if '&' not in line or line.startswith('\\'):
            continue
        parts = [p.strip() for p in re.split(r'(?<!\\)&', line.replace('\\\\', ''))]
        if len(parts) < 4:
            continue
        cluster = parts[0].replace('\\&', '&').strip()
        def parse_pct(s):
            m = re.search(r'([0-9]+\.[0-9]+)', s)
            return float(m.group(1)) if m else 0.0
        try:
            records.append({
                'Semantic Target Cluster': cluster,
                'Tree': parse_pct(parts[1]),
                'Deep': parse_pct(parts[2]),
                'Regularized Linear': parse_pct(parts[3]),
            })
        except Exception:
            continue

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError(f"Could not parse any rows from {TABLE_PATH}")

print(f"Loaded {len(df)} clusters from live table: {df['Semantic Target Cluster'].tolist()}")

ARCH_COLS = ['Tree', 'Deep', 'Regularized Linear']
ARCH_LABELS = {'Tree': 'Tree Ensembles', 'Deep': 'Deep Architectures', 'Regularized Linear': 'Linear Architectures'}
COLORS = {'Tree': '#b22222', 'Deep': '#1e90ff', 'Regularized Linear': '#228b22'}

df_melt = df.melt(id_vars='Semantic Target Cluster', value_vars=ARCH_COLS,
                  var_name='Architecture', value_name='Reliance')
df_melt['Architecture Label'] = df_melt['Architecture'].map(ARCH_LABELS)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
fig.suptitle("Comparative Primary Reliance: Top Feature Clusters Across Architectures",
             fontsize=14, fontweight='bold', y=1.02)

for ax, arch in zip(axes, ARCH_COLS):
    sub = df_melt[df_melt['Architecture'] == arch].sort_values('Reliance', ascending=True)
    sns.barplot(data=sub, x='Reliance', y='Semantic Target Cluster',
                color=COLORS[arch], alpha=0.85, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title(ARCH_LABELS[arch], fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel("Relative Attribution Share (%)", fontsize=10)
    ax.set_ylabel("")
    ax.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight', dpi=300)
print(f"[+] Saved Figure 10: {OUT_PDF}")
