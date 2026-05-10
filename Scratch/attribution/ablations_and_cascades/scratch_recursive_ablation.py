import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import seaborn as plt_sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
plt_sns.set_palette("husl")
plt_sns.set_context("talk")

DATA = r'C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\canonical'
ARTIFACT_DIR = r'C:\Users\dhl\.gemini\antigravity\brain\13179483-b897-4efa-b65b-6259cbe2174e'

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched_v2_OmniLagged.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
cat_cols = [c for c in df.columns if c.startswith('raw_')]
X_raw = df.drop(columns=[c for c in (drop_cols + future_features + cat_cols) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
X_raw = X_raw.loc[:, X_raw.var() > 0]
feature_names = np.array(X_raw.columns)
years = df['year'].values
y = df['is_protested'].values

# Establish the Domain Definitions (But don't force CatBoost to use them!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_scaled)
agg = FeatureAgglomeration(n_clusters=pca.n_components_, metric='euclidean', linkage='ward')
agg.fit(X_scaled)
cluster_labels = agg.labels_

print('Train Native Raw CatBoost (Access to all 801 Dimensions organically)...')
anchor = 2021
train_mask = years <= anchor

# CatBoost uniquely trained on RAW features natively!
model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=42)
model.fit(X_raw.values[train_mask], y[train_mask])

offset = 1
test_mask = years == (anchor + offset)
X_test_val = X_raw.values[test_mask]
y_test = y[test_mask]

# Get the native raw SHAP
shap_matrix = model.get_feature_importance(type='ShapValues', data=Pool(X_test_val, label=y_test))[:, :-1]
abs_mean_shap_raw = np.squeeze(np.mean(np.abs(shap_matrix), axis=0))

# Post-Execution: Aggregate exact RAW SHAPs computationally UP into the domain bounds algebraically
domain_shaps = np.zeros(pca.n_components_)
for cid in range(pca.n_components_):
    member_idx = np.where(cluster_labels == cid)[0]
    domain_shaps[cid] = np.sum(abs_mean_shap_raw[member_idx])

# Sort domains by highest topological SHAP accumulation
domain_ranking = np.argsort(domain_shaps)[::-1]

y_preds = model.predict_proba(X_test_val)[:, 1]
baseline_prauc = average_precision_score(y_test, y_preds)

ablation_results = [{'Features_Ablated': 0, 'PR_AUC': baseline_prauc, 'Degradation': 0.0}]

# Copy precisely to safely test topological domain degradation
X_test_ablated = np.copy(X_test_val)

print(f'\n[Recursive Physical OOD Degredation Log (Native Architecture)]')
print(f'Baseline Architecture: {baseline_prauc:.4f} PR-AUC')

# Evaluate strictly erasing the top K dimensions consecutively
steps_to_evaluate = list(range(1, 16))
for step in steps_to_evaluate:
    # Continuously scramble the entire group of physical features natively mapping to the mathematical group
    target_cluster_idx = domain_ranking[step - 1]
    member_idx = np.where(cluster_labels == target_cluster_idx)[0]
    
    # Ablate every single native feature organically inside this topological domain securely
    for f_idx in member_idx:
        np.random.shuffle(X_test_ablated[:, f_idx])
    
    y_preds_ablated = model.predict_proba(X_test_ablated)[:, 1]
    ablated_prauc = average_precision_score(y_test, y_preds_ablated)
    drop_pct = ((baseline_prauc - ablated_prauc) / baseline_prauc) * 100
    
    print(f'Erasing Domain C{target_cluster_idx+1} ({len(member_idx)} Native Sub-Features): {ablated_prauc:.4f} PR-AUC (-{drop_pct:.1f}%)')
    ablation_results.append({'Features_Ablated': step, 'PR_AUC': ablated_prauc, 'Degradation': drop_pct})

results_df = pd.DataFrame(ablation_results)

# Generate Lineplot exactly mapping explicitly
fig, ax = plt.subplots(figsize=(10, 6), facecolor="#121212")
fig.patch.set_facecolor('#121212')

ax.plot(results_df['Features_Ablated'], results_df['PR_AUC'], marker='o', color='#ff3366', linewidth=2.5, markersize=8)
ax.axhline(baseline_prauc, color='gray', linestyle='--', alpha=0.5, label='Baseline OOD P-RAUC')
ax.fill_between(results_df['Features_Ablated'], results_df['PR_AUC'], baseline_prauc, color='#ff3366', alpha=0.1)

ax.set_title("Native OOD Structural Entropy (True Raw Array Domain Ablation)", color='white', size=14, pad=15)
ax.set_xlabel("Number of Top Unsupervised Domains Systematically Ablated", color='lightgray', size=12)
ax.set_ylabel("Out-Of-Distribution Precision-Recall AUC", color='lightgray', size=12)
ax.tick_params(colors='lightgray')

ax.set_xticks(range(0, 16, 2))
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(color='gray', linestyle=':', alpha=0.2)
ax.legend(facecolor='#121212', edgecolor='gray', labelcolor='lightgray')

plt.tight_layout()
out_path = os.path.join(ARTIFACT_DIR, 'plot_final_native_ablation_degradation.png')
fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')

print(f'\n[*] Recursive array sequence plotted securely dynamically securely to: {out_path}')
