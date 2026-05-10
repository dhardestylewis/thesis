import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set visual aesthetics based on the user's prior guidelines
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

model_order = ['Random Forest', 'CatBoost', 'XGBoost', 'LightGBM', 'Logistic Regression']
palette = {'Random Forest': '#00BFA5', 'CatBoost': '#00E5FF', 'XGBoost': '#2979FF', 'LightGBM': '#651FFF', 'Logistic Regression': '#FF1744'}

# Filter to positive offsets (OOD only)
offset_df = df[df['Offset'] >= 0].copy()

# -------------------------------------------------------------
# Offset Degradation Lineplot (Performance by Offset)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7), facecolor="#121212")
ax.set_facecolor("#121212")

# We use sns.lineplot with errorbar='ci' or just plotting the mean natively across all anchors/seeds for that offset
sns.lineplot(
    data=offset_df, x='Offset', y='PRAUC', hue='Model', 
    hue_order=model_order, palette=palette, 
    marker='o', markersize=8, linewidth=3, err_style='band', errorbar=('ci', 95), ax=ax
)

ax.set_title("OOD Performance Degradation by Temporal Horizon (+N Years)", fontsize=16, color='white', pad=20)
ax.set_xlabel("Years Since Training Anchor (Offset)", fontsize=12, color='white')
ax.set_ylabel("Mean PR-AUC (with 95% CI)", fontsize=12, color='white')
ax.tick_params(colors='lightgray')
ax.grid(True, color='#333', linestyle='--')

ax.legend(facecolor='#1A1A1A', edgecolor='#333', labelcolor='white')
sns.despine(left=True, bottom=True)
plt.tight_layout()

out_offset = os.path.join(ARTIFACT_DIR, 'plot_final_offset_degradation.png')
fig.savefig(out_offset, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=False)
plt.close(fig)

print(f"[{out_offset}] saved successfully.")
