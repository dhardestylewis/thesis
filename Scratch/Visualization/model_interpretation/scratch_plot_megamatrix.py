import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DRAFT_DIR = os.path.join(ROOT, "Thesis_Draft")
df = pd.read_csv(os.path.join(DRAFT_DIR, "Mega_Matrix_Full_Results.csv"))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
palette = sns.color_palette("husl", 4)

# Plot 1: Geometric Decay by Offset
plt.figure(figsize=(10, 6))
offset_df = df[df['Geographic_Layer'] != 'Absolute']
sns.lineplot(data=offset_df, x='OOS_Offset', y='MdAPE', hue='Geographic_Layer', marker='o', linewidth=2.5, palette=palette[:3], ci=None)
plt.title("Spatial ML Failure: Error Decay by OOT Offset Horizon", fontsize=14, weight='bold')
plt.xlabel("Out-Of-Time Evaluation Offset", fontsize=12)
plt.ylabel("Median Absolute Percentage Error (MdAPE)", fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.legend(title='Spatial Boundary')
plt.tight_layout()
plt.savefig(os.path.join(DRAFT_DIR, "plot_offset_decay.png"), dpi=300)
plt.close()

# Plot 2: Anchor Stability (Tracking chronologically)
plt.figure(figsize=(12, 6))
anchor_df = df.groupby(['Anchor_Year', 'Geographic_Layer'])['MdAPE'].mean().reset_index()
sns.lineplot(data=anchor_df, x='Anchor_Year', y='MdAPE', hue='Geographic_Layer', style='Geographic_Layer', markers=True, dashes=False, linewidth=2.5, palette=palette)
plt.title("Universal Topological Collapse Across All Timelines (2018-2024)", fontsize=14, weight='bold')
plt.xlabel("Training Anchor (Pre-Year)", fontsize=12)
plt.ylabel("Average Forward MdAPE", fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.legend(title='Geographic Limit', loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(DRAFT_DIR, "plot_anchor_stability.png"), dpi=300)
plt.close()

# Plot 3: Architectural Parity (Boxplot proving all models fail locally)
plt.figure(figsize=(12, 6))
# Only focus on Micro (Census Tract) and Meso (Zipcode) to prove they fail across ALL hyperparams
sub_df = df[df['Geographic_Layer'].isin(['Micro', 'Meso'])]
sns.boxplot(data=sub_df, x='Architecture', y='MdAPE', hue='Geographic_Layer', palette="Set2")
plt.title("Architectural Parity: Absolute Failure Across Regressor Formulations", fontsize=14, weight='bold')
plt.xlabel("Algorithm Configuration", fontsize=12)
plt.ylabel("Error Distribution (MdAPE) in Micro/Meso Bounds", fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.axhline(0.5, ls='--', color='red', label='50% Error Threshold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DRAFT_DIR, "plot_architectural_parity.png"), dpi=300)
plt.close()

print("Plots successfully rendered and dumped to Draft Directory.")
