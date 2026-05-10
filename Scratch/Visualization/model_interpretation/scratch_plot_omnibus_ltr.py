import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DRAFT_DIR = os.path.join(ROOT, "Thesis_Draft")
df = pd.read_csv(os.path.join(DRAFT_DIR, "Omnibus_LTR_Matrix.csv"))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Plot 1: The LTR Stacking Poison 
plt.figure(figsize=(12, 6))
t_order = ['Base_Ranker', 'Meta_Ranker']
sns.boxplot(data=df, x='Offset', y='Score', hue='Topology', hue_order=t_order, palette="autumn")
plt.title("Relational LTR Poison: Structural NDCG Degradation via Continuous Stacking", fontsize=14, weight='bold')
plt.xlabel("Evaluation Offset Timeline", fontsize=12)
plt.ylabel("Testing Relevance (NDCG)", fontsize=12)
plt.legend(title='Architecture Topology', loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(DRAFT_DIR, "plot_omnibus_ltr_poison.png"), dpi=300)
plt.close()

# Plot 2: LTR Target Binning Parity
plt.figure(figsize=(12, 6))
# Filter to just Base Ranker to show native performance
base_df = df[df['Topology'] == 'Base_Ranker'].copy()
sns.boxplot(data=base_df, x='Offset', y='Score', hue='Target_Binning', palette="winter")
plt.title("Relational Parity: LTR mapping natively agnostic to target continuous vs categorical limits", fontsize=14, weight='bold')
plt.xlabel("Evaluation Offset Timeline", fontsize=12)
plt.ylabel("Native Baseline Relevance (NDCG)", fontsize=12)
plt.legend(title='Target Formulation', loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(DRAFT_DIR, "plot_omnibus_ltr_target_parity.png"), dpi=300)
plt.close()

print("LTR Boxplots successfully rendered and dumped to Draft Directory.")
