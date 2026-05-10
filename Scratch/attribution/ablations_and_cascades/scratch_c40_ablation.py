import pandas as pd
import numpy as np
import os
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

DATA = r'C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\canonical'
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_scaled)
agg = FeatureAgglomeration(n_clusters=pca.n_components_, metric='euclidean', linkage='ward')
X_clustered_raw = agg.fit_transform(X_scaled)
cluster_labels = agg.labels_

print('Calculating 1-Anchor test constraint specifically for C40 (Cluster 40) natively...')
anchor = 2021
train_mask = years <= anchor
model = CatBoostClassifier(iterations=100, depth=6, verbose=0, random_seed=42)
model.fit(X_clustered_raw[train_mask], y[train_mask])

offset = 1
test_mask = years == (anchor + offset)
X_test_val = X_clustered_raw[test_mask]
y_test = y[test_mask]

shap_matrix = model.get_feature_importance(type='ShapValues', data=Pool(X_test_val, label=y_test))[:, :-1]
abs_mean_shap = np.squeeze(np.mean(np.abs(shap_matrix), axis=0))

# 1. Relative Global SHAP calculation natively:
total_shap_units = np.sum(abs_mean_shap)
c40_shap = abs_mean_shap[39] # C40 is index 39 (0-indexed)
relative_pct = (c40_shap / total_shap_units) * 100

# 2. Exact PR-AUC Ablation (Lost PR-AUC if organically mapped randomly via Permutation natively)
y_preds = model.predict_proba(X_test_val)[:, 1]
baseline_prauc = average_precision_score(y_test, y_preds)

X_test_ablated = np.copy(X_test_val)
np.random.shuffle(X_test_ablated[:, 39]) # Randomly scramble Cluster 40 organically
y_preds_ablated = model.predict_proba(X_test_ablated)[:, 1]
ablated_prauc = average_precision_score(y_test, y_preds_ablated)

print(f'\n[SHAP Relativity]')
print(f'Total Model SHAP Effort Allocated Across All {pca.n_components_} Domains: {total_shap_units:.4f}')
print(f'Effort Monopolized solely by C40: {c40_shap:.4f} ({relative_pct:.2f}% of entire model cognition)')

print(f'\n[Ablation Physical PR-AUC Loss]')
print(f'True OOD Performance linearly: {baseline_prauc:.4f} PR-AUC')
print(f'Performance if C40 structural data is randomized (Permutation Ablation): {ablated_prauc:.4f} PR-AUC')
print(f'Total Absolute Predictive Degredation: {baseline_prauc - ablated_prauc:.4f}')
print(f'Global Relative Accuracy Erased: {((baseline_prauc - ablated_prauc) / baseline_prauc)*100:.2f}%!')
