import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
X_raw = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
y = df['is_protested'].values
years = df['year'].values

anchors = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

drift_results = []
for anchor in anchors:
    train_mask = years < anchor
    if train_mask.sum() < 50: continue
    X_train_raw, y_train = X_raw.values[train_mask], y[train_mask]
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)

    models = {
        'CatBoost': CatBoostClassifier(iterations=100, depth=6, verbose=0, random_seed=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Logistic': LogisticRegression(class_weight='balanced', random_state=42)
    }

    for name, m in models.items():
        if name == 'Logistic': m.fit(X_train_sc, y_train)
        else: m.fit(X_train_raw, y_train)

    for test_year in eval_years:
        if test_year < anchor: continue
        test_mask = years == test_year
        if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
            
        X_test_raw, y_test = X_raw.values[test_mask], y[test_mask]
        X_test_sc = scaler.transform(X_test_raw)
        
        for name, m in models.items():
            p = m.predict_proba(X_test_sc if name == 'Logistic' else X_test_raw)[:, 1]
            prauc = average_precision_score(y_test, p)
            drift_results.append({
                'Model': name, 'Anchor': f'Pre-{anchor}',
                'Evaluate_Year': test_year, 
                'Offset': test_year - anchor, 
                'PRAUC': prauc
            })

res = pd.DataFrame(drift_results)
OUT_DIR = os.path.join(ROOT, 'Thesis_Draft', 'Draft_v1', 'Track1_Exhibits')

# PLOT 1: Anchor-year trajectories across holdout years (CatBoost only)
cb_data = res[res['Model'] == 'CatBoost'].copy()
plt.figure(figsize=(7, 5))
sns.lineplot(data=cb_data, x='Offset', y='PRAUC', hue='Anchor', marker='o', linewidth=2.5, palette='viridis')
plt.title('Temporal Predictive Drift (Rolling Origin) - CatBoost')
plt.xlabel('Years Out-of-Distribution (T + offset)')
plt.ylabel('PR-AUC')
plt.ylim(0, 1.05)
plt.axhline(y=0.057, color='r', linestyle='--', label='Pooled Baseline (~6%)')
plt.legend(title='Training Anchor', loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_temporal_drift_H0.pdf'), format='pdf', dpi=300)
plt.close()

# PLOT 2: Holdout-year comparison view (Bar chart)
# Let's pick a specific evaluation year representing a distant "Holdout", e.g. 2022. 
# Show how different anchors perform at predicting 2022.
# Wait, let's just make it a barplot grouping by Model, colored by Anchor, for predict-next-year (Offset=0).
# Even better: The original plot was "Model Rot over Time", showing pre-2019, pre-2020, pre-2021 predicting some horizon.
# Let's show average PR-AUC across all horizons for each model/anchor, or just select 3 anchors (Pre-2018, Pre-2020, Pre-2022) to show the drop.
sub_anchors = ['Pre-2018', 'Pre-2020', 'Pre-2022']
bp_data = res[res['Anchor'].isin(sub_anchors) & (res['Offset'] >= 0)]
# Calculate mean PR-AUC across whatever horizons they lived to see
bp_agg = bp_data.groupby(['Model', 'Anchor'])['PRAUC'].mean().reset_index()

plt.figure(figsize=(7, 5))
sns.barplot(data=bp_agg, x='Model', y='PRAUC', hue='Anchor', palette='magma')
plt.title('Predictive Temporal Drift (Average Out-Of-Sample)')
plt.xlabel('Model')
plt.ylabel('Average Out-Of-Sample PR-AUC')
plt.ylim(0, 1.0)
plt.axhline(y=0.057, color='r', linestyle='--', label='Pooled Random Chance (~6%)')
plt.legend(title='Training Anchor', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_temporal_drift.pdf'), format='pdf', dpi=300)
plt.close()

print("Figures successfully generated!")
