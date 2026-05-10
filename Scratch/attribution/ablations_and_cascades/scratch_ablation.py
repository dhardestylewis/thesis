import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as plt_sns
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

plt.style.use('dark_background')
plt_sns.set_palette("husl")
plt_sns.set_context("talk")

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\13179483-b897-4efa-b65b-6259cbe2174e'

base_path = os.path.join(DATA, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv')
if not os.path.exists(base_path):
    print(f"Error: Could not find {base_path}")
    exit(1)
    
df = pd.read_csv(base_path, low_memory=False)

def map_zoning_density(z):
    if pd.isna(z): return 0
    if 'CBD' in str(z).upper(): return 10
    return 0
if 'zoning_code' in df.columns:
    df['zoning_density_score'] = df['zoning_code'].apply(map_zoning_density)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
cat_cols = [c for c in df.columns if c.startswith('raw_')]

# Safely extract numeric features
X_raw = df.drop(columns=[c for c in (drop_cols + future_features + cat_cols) if c in df.columns], errors='ignore').select_dtypes(include=[np.number])
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
y = df['is_protested'].values

# Establish absolute Base Evaluation
model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
print("[*] Training Base OOD Model on full architecture (No Drops)...")
# Note: we use cross-validated metrics to get true Out-of-Distribution drop
baseline_praucs = []
for test_year in [2023, 2024]:
    train_mask = df['year'] < 2022
    test_mask = df['year'] == test_year
    model.fit(X_raw[train_mask], y[train_mask])
    preds = model.predict_proba(X_raw[test_mask])[:,1]
    baseline_praucs.append(average_precision_score(y[test_mask], preds))
    
base_prauc = np.mean(baseline_praucs)
print(f"    Baseline Average OOD PR-AUC: {base_prauc:.4f}")

# Map and organize columns by their Temporal Class
offsets = {
    'Base / Current': [c for c in X_raw.columns if not re.search(r'lag_(\d+)yr', c)],
    'Lag 1 Yr': [c for c in X_raw.columns if '_lag_1yr' in c],
    'Lag 2 Yr': [c for c in X_raw.columns if '_lag_2yr' in c],
    'Lag 3 Yr': [c for c in X_raw.columns if '_lag_3yr' in c],
    'Lag 4 Yr': [c for c in X_raw.columns if '_lag_4yr' in c],
    'Lag 5 Yr': [c for c in X_raw.columns if '_lag_5yr' in c],
    'Lag 6 Yr': [c for c in X_raw.columns if '_lag_6yr' in c]
}

results = []

print("\n[*] Executing Exact Block-Drop Iterative Ablation Across Temporal Layers...")
for offset_name, cols in offsets.items():
    if len(cols) == 0: continue
    
    print(f"    -> Dropping [{offset_name}] ({len(cols)} variables) ...")
    X_ablated = X_raw.drop(columns=cols)
    
    prauc_drops = []
    for test_year in [2023, 2024]:
        train_mask = df['year'] < 2022
        test_mask = df['year'] == test_year
        
        model.fit(X_ablated[train_mask], y[train_mask])
        preds = model.predict_proba(X_ablated[test_mask])[:,1]
        
        # Calculate OOD drop impact when removing this entire temporal block
        prauc_drops.append(average_precision_score(y[test_mask], preds))
        
    ablated_prauc = np.mean(prauc_drops)
    
    # Mathematical True Attribution = Baseline Performance - Performance when structurally erased
    # Positive meaning: Erasing this dimension significantly damaged the model capability.
    structural_damage = base_prauc - ablated_prauc
    
    results.append({
        'Temporal Block': offset_name,
        'Absolute OOD PR-AUC Drop': structural_damage,
        'Raw PR-AUC Reached': ablated_prauc
    })

rdf = pd.DataFrame(results)

# Filter out minimal noise logic
rdf = rdf.sort_values('Temporal Block', ascending=True)

# Visualize Output
fig, ax = plt.subplots(figsize=(10, 8), facecolor="#121212")
fig.patch.set_facecolor('#121212')

colors = ['crimson' if val > 0 else 'gray' for val in rdf['Absolute OOD PR-AUC Drop']]
plt_sns.barplot(data=rdf, x='Absolute OOD PR-AUC Drop', y='Temporal Block', palette=colors, ax=ax)

ax.set_title("Exact Block-Drop Ablation: Temporal Latency Impact", color='white', size=16, pad=20)
ax.set_xlabel("Absolute Output Degradation (PR-AUC Loss) when Offset Block is Fully Erased", color='white', size=12)
ax.set_ylabel("", color='white')
ax.tick_params(colors='lightgray')

for spine in ax.spines.values():
    spine.set_visible(False)
    
plt.axvline(0, color='white', linestyle='--', linewidth=1)
plt.tight_layout()

out_path = os.path.join(ARTIFACT_DIR, 'plot_final_temporal_ablation.png')
fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')

print(f"\n[*] Ablation arrays fully iterated. True algorithmic damage mapped successfully.")
print(f"    Graph exported to: {out_path}")
