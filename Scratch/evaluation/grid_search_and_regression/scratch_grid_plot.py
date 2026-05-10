import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
res_df = pd.read_csv(os.path.join(ROOT, 'grid_tuning_results_full.csv'))

universal_mean = res_df.groupby(['Model', 'Profile'])['PRAUC'].mean().reset_index()

sns.set_theme(style='whitegrid')
plt.figure(figsize=(12, 6))
g = sns.barplot(
    data=universal_mean,
    x='Model',
    y='PRAUC',
    hue='Profile',
    palette='viridis'
)

plt.title('Universal Mean PR-AUC across All Anchor Constraints (60 Models)', fontsize=14, fontweight='bold')
plt.ylabel('OOS Universal PR-AUC', fontsize=12)
plt.xlabel('Architectural Family', fontsize=12)
plt.legend(title='Hyperparameter Profile', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
artifact_dir = r'C:\Users\dhl\.gemini\antigravity\brain\f177875b-a899-4360-bdeb-38a69114ef25'
plt.savefig(os.path.join(artifact_dir, 'universal_tuning_plot.png'), dpi=300, bbox_inches='tight')
print("[*] Plot saved to artifacts.")
