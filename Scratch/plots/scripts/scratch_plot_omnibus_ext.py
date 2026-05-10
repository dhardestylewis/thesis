import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DRAFT_DIR = os.path.join(ROOT, "Thesis_Draft")
try:
    df = pd.read_csv(os.path.join(DRAFT_DIR, "Omnibus_LTR_Matrix_Extreme.csv"))
except FileNotFoundError:
    print("Could not find the extreme CSV.")
    exit(1)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Extract depth from architecture name for clearer plotting
def get_depth(row):
    if 'Depth10' in row['Architecture']: return 'Depth 10'
    return 'Depth 6'
df['Depth'] = df.apply(get_depth, axis=1)

# Plot 1: The Relative Decay + Over-fitting Metric (Offset 1 -> 6)
plt.figure(figsize=(14, 6))
base_df = df[df['Topology'] == 'Base_Ranker'].copy()
sns.boxplot(data=base_df, x='Offset', y='Score', hue='Depth', palette="plasma")
plt.title("Longitudinal Limit: Relational Hierarchy Decay across Deep-Network Constraints (+6Yr)", fontsize=14, weight='bold')
plt.xlabel("Evaluation Offset Timeline", fontsize=12)
plt.ylabel("Native Baseline Relevance (NDCG)", fontsize=12)
plt.legend(title='Testing Depth', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(DRAFT_DIR, "plot_extreme_ltr_decay.png"), dpi=300)
plt.close()

# Plot 2: The Extended Poisson Protocol (Expanded to +6 yrs)
plt.figure(figsize=(14, 6))
# Filter specifically strictly to depth 6 to see pure architectural poison mapping
d6_df = df[df['Depth'] == 'Depth 6'].copy()
t_order = ['Base_Ranker', 'Meta_Ranker']
sns.boxplot(data=d6_df, x='Offset', y='Score', hue='Topology', hue_order=t_order, palette="cool")
plt.title("Extended Topography Poison: NDCG Degradation via Continuous Stacking Matrix (+6Yr Limit)", fontsize=14, weight='bold')
plt.xlabel("Evaluation Offset Timeline", fontsize=12)
plt.ylabel("Testing Relevance (NDCG)", fontsize=12)
plt.legend(title='Architecture Topology', loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(DRAFT_DIR, "plot_extreme_ltr_poison.png"), dpi=300)
plt.close()

print("Extreme LTR Boxplots successfully rendered and dumped to Draft Directory.")
