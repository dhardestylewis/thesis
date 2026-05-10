import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set visual aesthetics based on the user's prior guidelines (vibrant/dark aesthetics)
plt.style.use('dark_background')
sns.set_palette("husl")
sns.set_context("talk", rc={"axes.grid": True, "grid.color": "#333", "grid.linestyle": "--"})

DRAFT_DIR = r'C:\Users\dhl\data\thesis\thesis\Thesis_Draft'
ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\13179483-b897-4efa-b65b-6259cbe2174e'

csv_path = os.path.join(DRAFT_DIR, 'Multiseed_PRAUC_Table5_Validation.csv')
if not os.path.exists(csv_path):
    print("Error: Results CSV not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Ensure models are appropriately ordered for aesthetic
model_order = ['Random Forest', 'CatBoost', 'XGBoost', 'LightGBM', 'Logistic Regression']
palette = {'Random Forest': '#00BFA5', 'CatBoost': '#00E5FF', 'XGBoost': '#2979FF', 'LightGBM': '#651FFF', 'Logistic Regression': '#FF1744'}

# -------------------------------------------------------------
# 1. Headline OOD Summary Boxplot (Years 2023-2024, Anchor <= 2022)
# -------------------------------------------------------------
ood_df = df[(df['Evaluate_Year'] >= 2023) & (df['Anchor'] <= 2022)].copy()

fig, ax = plt.subplots(figsize=(10, 6), facecolor="#121212")
ax.set_facecolor("#121212")
sns.boxplot(data=ood_df, x='PRAUC', y='Model', order=model_order, palette=palette, ax=ax, boxprops=dict(alpha=0.6), fliersize=0)
sns.stripplot(data=ood_df, x='PRAUC', y='Model', order=model_order, palette=palette, ax=ax, size=6, alpha=0.8, jitter=0.2)

ax.set_title("OOD Out-of-Sample PR-AUC Tracking (2023-2024 Holdouts)", fontsize=16, color='white', pad=20)
ax.set_xlabel("PR-AUC Lift Validation Score", fontsize=12, color='white')
ax.set_ylabel("", fontsize=12, color='white')
ax.tick_params(axis='x', colors='lightgray')
ax.tick_params(axis='y', colors='lightgray')
ax.grid(axis='x', color='#333', linestyle='--')

sns.despine(left=True, bottom=True)
plt.tight_layout()

out_ood = os.path.join(ARTIFACT_DIR, 'plot_final_ood_boxplot.png')
fig.savefig(out_ood, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
plt.close(fig)

# -------------------------------------------------------------
# 2. Sequential Validation Degradation (Lineplot Across Eval Years)
# -------------------------------------------------------------
# To keep the plot clean, we will plot the Mean PR-AUC per Model across Evaluation Years (evaluating how they handle drift)
time_df = df.groupby(['Evaluate_Year', 'Model'])['PRAUC'].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(12, 7), facecolor="#121212")
ax2.set_facecolor("#121212")

for model in model_order:
    sub = time_df[time_df['Model'] == model].sort_values('Evaluate_Year')
    ax2.plot(sub['Evaluate_Year'], sub['PRAUC'], label=model, color=palette[model], linewidth=3, marker='o', markersize=8)

ax2.set_title("Longitudinal Spatial Model Degradation Over Time", fontsize=16, color='white', pad=20)
ax2.set_xlabel("Evaluation Year", fontsize=12, color='white')
ax2.set_ylabel("Mean PR-AUC", fontsize=12, color='white')
ax2.tick_params(colors='lightgray')
ax2.grid(True, color='#333', linestyle='--')

# Highlight the out-of-distribution jump point visually
ax2.axvspan(2022.5, 2024.5, color='#FF1744', alpha=0.1, label='Extreme OOD Phase')

ax2.legend(facecolor='#1A1A1A', edgecolor='#333', labelcolor='white')
sns.despine(left=True, bottom=True)
plt.tight_layout()

out_deg = os.path.join(ARTIFACT_DIR, 'plot_final_degradation_lineplot.png')
fig2.savefig(out_deg, dpi=300, bbox_inches='tight', facecolor=fig2.get_facecolor(), transparent=False)
plt.close(fig2)

print("\nSaved visuals to artifact directory:")
print(f"  {out_ood}")
print(f"  {out_deg}")
