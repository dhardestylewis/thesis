import pandas as pd, numpy as np, os, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from catboost import CatBoostRanker, Pool
from xgboost import XGBRanker
from lightgbm import LGBMRanker
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['council_district'] = df['council_district'] if 'council_district' in df.columns else df.get('council_district_x', 1)
df['council_district'] = df['council_district'].fillna(1).astype(str)

df = df.dropna(subset=['year', 'is_protested', 'case_number'])

try:
    pet = pd.read_csv(os.path.join(ROOT, "Data", "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv"))
    df = df.merge(pet[['case_number', 'signer_pct']], on='case_number', how='left')
    df['signed_area_share'] = (df['signer_pct'] / 100.0).fillna(0.0)
except Exception as e:
    raise RuntimeError("Missing Petition file for continuous analysis.")

# Create Query Groups
le = LabelEncoder()
df['group_id_str'] = df['council_district'].astype(str) + "_" + df['year'].astype(str)
df['group_id'] = le.fit_transform(df['group_id_str'])

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share', 'group_id_str', 'group_id']
fut_feat = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']

X_raw_df = df.drop(columns=[c for c in (drop_cols + fut_feat) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

# Discretization Filter
phys_floats = ['ldb_appraised_val', 'land_market_value', 'total_market_value', 'gross_site_area_acres', 'deed_acreage', 'ldb_land_acres', 'ldb_lotsize', 'improvement_sq_ft', 'ldb_imprv_sqft']
to_discretize = [c for c in phys_floats if c in X_raw_df.columns]
for col in to_discretize:
    dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    X_col = X_raw_df[[col]].values
    dt.fit(X_col, df['is_protested'].values)
    X_raw_df[col] = dt.apply(X_col)

# Strict sorting required for LTR algorithms
df['original_idx'] = np.arange(len(df))
df = df.sort_values(by=['group_id'])
sort_idx = df['original_idx'].values
X_raw = X_raw_df.values[sort_idx]

years = df['year'].values
y_reg = (df['signed_area_share'].values * 10).astype(int) # Int constraint for XGBRanker/LGBMRanker
groups = df['group_id'].values

# Architecture Configurations
def get_ranker(algo_name, depth):
    if algo_name == 'CatBoost':
        return CatBoostRanker(iterations=100, depth=depth, loss_function='YetiRank', random_seed=42, verbose=0)
    elif algo_name == 'XGBoost':
        return XGBRanker(n_estimators=100, max_depth=depth, random_state=42, objective='rank:ndcg', tree_method='hist')
    elif algo_name == 'LightGBM':
        return LGBMRanker(n_estimators=100, max_depth=depth, random_state=42, objective='lambdarank', verbose=-1)

models_conf = [
    ('CatBoost', 6, 'CatBoost_Default'),
    ('CatBoost', 10, 'CatBoost_HighCap'),
    ('CatBoost', 14, 'CatBoost_ExtDeep'),
    ('XGBoost', 6, 'XGB_Default'),
    ('XGBoost', 10, 'XGB_HighCap'),
#    ('XGBoost', 14, 'XGB_ExtDeep'), # extremely slow mathematically for XGB gpu rank tree
    ('LightGBM', 6, 'LGBM_Default'),
    ('LightGBM', 10, 'LGBM_HighCap'),
    ('LightGBM', 14, 'LGBM_ExtDeep')
]

anchors = [2018, 2019, 2020, 2021, 2022]
results = []

print(f"[*] Executing Universal LTR Matrix on {len(models_conf)} profiles across {len(anchors)} Anchors...")

for anchor in anchors:
    print(f"\n---> Anchor: Pre-{anchor}")
    train_mask = years < anchor
    X_train = X_raw[train_mask]
    y_train = y_reg[train_mask]
    g_train = groups[train_mask]
    
    # Calculate group sizes array for XGB/LGBM
    _, group_counts = np.unique(g_train, return_counts=True)
    
    for (algo, depth, profile) in models_conf:
        m = get_ranker(algo, depth)
        
        # Fit logic
        if algo == 'CatBoost':
            m.fit(X_train, y_train, group_id=g_train)
        else:
            m.fit(X_train, y_train, group=group_counts)
            
        # OOS Evaluation
        for test_year in range(anchor, 2025):
            test_mask = years == test_year
            if test_mask.sum() == 0: continue
                
            X_test, y_test_r, g_test = X_raw[test_mask], y_reg[test_mask], groups[test_mask]
            
            # Predict
            preds = m.predict(X_test)
            
            # Calculate Mean NDCG across independent groups
            ndcgs = []
            for g in np.unique(g_test):
                idx = (g_test == g)
                if y_test_r[idx].sum() > 0 and len(y_test_r[idx]) > 1:
                    ndcg = ndcg_score([y_test_r[idx]], [preds[idx]])
                    ndcgs.append(ndcg)
                    
            mean_ndcg = np.mean(ndcgs) if len(ndcgs) > 0 else np.nan
            
            results.append({
                'Profile': profile,
                'Anchor': anchor,
                'Evaluate_Year': test_year,
                'NDCG': mean_ndcg
            })

res_df = pd.DataFrame(results).dropna()
res_df.to_csv(os.path.join(ROOT, 'Analysis', 'Output', 'universal_ltr_matrix.csv'), index=False)

# Visual Plotting
plot_df = res_df.groupby(['Profile', 'Anchor'])['NDCG'].mean().reset_index()

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
plt.figure(figsize=(14, 8))

# Map specific visually distinguishable colors
palette = {
    'CatBoost_Default': '#00BFFF', 'CatBoost_HighCap': '#1E90FF', 'CatBoost_ExtDeep': '#0000FF',
    'XGB_Default': '#32CD32', 'XGB_HighCap': '#228B22', 'XGB_ExtDeep': '#006400',
    'LGBM_Default': '#FF8C00', 'LGBM_HighCap': '#FF4500', 'LGBM_ExtDeep': '#DC143C'
}

ax = sns.barplot(data=plot_df, x='Anchor', y='NDCG', hue='Profile', palette=palette, edgecolor='black', linewidth=1)
plt.title('Universal LTR Domain Drift Survival (Mean NDCG Across Forward Years)', fontsize=16, weight='bold', color='white', pad=20)
plt.xlabel('Temporal Training Anchor (Pre-Year)', fontsize=13, weight='bold')
plt.ylabel('Mean Out-of-Sample NDCG', fontsize=13, weight='bold')

plt.axhline(0.60, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Ranking Stability Tier')

plt.legend(title='Architecture Profile', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, facecolor='#2C2C2C', edgecolor='white', labelcolor='white')
plt.tight_layout()

art_path = r'C:\Users\dhl\.gemini\antigravity\brain\f177875b-a899-4360-bdeb-38a69114ef25\universal_ltr_plot.png'
plt.savefig(art_path, dpi=300, bbox_inches='tight')
print(f"[*] Plotted Universal LTR Matrix to: {art_path}")

print("\n=== LTR Mean NDCG Over Drift by Architecture ===")
agg = res_df.groupby(['Profile', 'Anchor'])['NDCG'].mean().unstack()
print(agg.round(3).to_markdown())

