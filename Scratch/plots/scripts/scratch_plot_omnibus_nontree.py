import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DRAFT_DIR = os.path.join(ROOT, "Thesis_Draft")
df = pd.read_csv(os.path.join(DRAFT_DIR, "Omnibus_Nontree_Matrix.csv"))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Plot 1: The Regression Stacking Safety Net (Non-Tree)
plt.figure(figsize=(12, 6))
reg_df = df[df['Target_Binning'] == 'Absolute_Continuous'].copy()
base_meta_order = ['Base_Regression', 'Meta_Regression']
sns.boxplot(data=reg_df, x='Offset', y='Score', hue='Topology', hue_order=base_meta_order, palette="Purples")
plt.title("Deep Learning/Linear Rescue: Meta-Stacking continuous error across General Manifolds", fontsize=14, weight='bold')
plt.xlabel("Evaluation Offset Timeline", fontsize=12)
plt.ylabel("Testing Error Distribution (MdAPE)", fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.axhline(0.5, ls='--', color='red', alpha=0.9, label='50% Error Benchmark')
plt.legend(title='Architecture Topology', loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(DRAFT_DIR, "plot_nontree_regression_rescue.png"), dpi=300)
plt.close()

# Plot 2: The Classifier Stacking Poison (Non-Tree)
plt.figure(figsize=(12, 6))
cls_df = df[df['Target_Binning'] == 'Boolean_Legal'].copy()
c_order = ['Base_Classifier', 'Meta_Classifier']
sns.boxplot(data=cls_df, x='Offset', y='Score', hue='Topology', hue_order=c_order, palette="Greens")
plt.title("Deep Learning/Linear Poison: Structural degradation of Neural Net Thresholds via Stacking", fontsize=14, weight='bold')
plt.xlabel("Evaluation Offset Timeline", fontsize=12)
plt.ylabel("Precision-Recall AUC (PR-AUC)", fontsize=12)
plt.legend(title='Architecture Topology', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(DRAFT_DIR, "plot_nontree_classifier_poison.png"), dpi=300)
plt.close()

print("Non-Tree Boxplots successfully rendered and dumped to Draft Directory.")
