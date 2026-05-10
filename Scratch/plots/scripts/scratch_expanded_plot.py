import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
res_df = pd.read_csv(os.path.join(ROOT, 'grid_tuning_results_expanded.csv'))

universal_mean = res_df.groupby(['Model', 'Profile'])['PRAUC'].mean().reset_index()

sns.set_theme(style='whitegrid')
plt.figure(figsize=(16, 8))
# Order by model family to make grouping clear
model_order = ['RandomForest', 'XGBoost', 'CatBoost', 'LightGBM', 'Logistic', 'TabNet', 'TabNetVREx']
profile_order = ['ExtShallow', 'Regularized', 'Default', 'HighCap', 'ExtDeep']

g = sns.barplot(
    data=universal_mean,
    x='Model',
    y='PRAUC',
    hue='Profile',
    palette='magma',
    order=model_order,
    hue_order=profile_order
)

plt.title('Ultimate Universal Tuning Grid: 175 Model Extrapolation across 5 Temporal Anchors', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('OOS Universal PR-AUC (Multi-Anchor Average)', fontsize=14)
plt.xlabel('Architectural Base', fontsize=14)
plt.legend(title='Hyperparameter Capacity Profile', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

# Add horizontal line for the baseline mean
baseline = universal_mean['PRAUC'].mean()
plt.axhline(baseline, color='red', linestyle='--', alpha=0.5, label='Global PRAUC Mean')

plt.tight_layout()
artifact_dir = r'C:\Users\dhl\.gemini\antigravity\brain\f177875b-a899-4360-bdeb-38a69114ef25'
plt.savefig(os.path.join(artifact_dir, 'ultimate_tuning_plot.png'), dpi=300, bbox_inches='tight')
print("[*] Ultimate Plot saved to artifacts.")
