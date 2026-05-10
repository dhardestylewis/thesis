import pandas as pd, numpy as np, os
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier

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
        'CatBoost (Trees)': CatBoostClassifier(iterations=100, depth=6, verbose=0, random_seed=42),
        'XGBoost (Trees)': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'LightGBM (Trees)': LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1),
        'Random Forest (Trees)': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Logistic L2 (Linear)': LogisticRegression(class_weight='balanced', random_state=42)
    }

    for name, m in models.items():
        if 'Linear' in name: m.fit(X_train_sc, y_train)
        else: m.fit(X_train_raw, y_train)

    for test_year in eval_years:
        if test_year < anchor: continue
        test_mask = years == test_year
        if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
            
        X_test_raw, y_test = X_raw.values[test_mask], y[test_mask]
        X_test_sc = scaler.transform(X_test_raw)
        
        for name, m in models.items():
            p = m.predict_proba(X_test_sc if 'Linear' in name else X_test_raw)[:, 1]
            prauc = average_precision_score(y_test, p)
            family = 'Linear' if 'Linear' in name else 'Trees'
            drift_results.append({
                'Family': family, 'Model': name, 'Anchor': f'Pre-{anchor}',
                'Evaluate_Year': test_year, 'PRAUC': prauc
            })

results_df = pd.DataFrame(drift_results)
pivot = results_df.groupby(['Family', 'Anchor', 'Evaluate_Year'])['PRAUC'].max().reset_index()
pivot_mat = pivot.pivot_table(index='Family', columns=['Anchor', 'Evaluate_Year'], values='PRAUC')
winners = pivot_mat.idxmax()

tex_lines = [
    "| Anchor | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
]

for anchor in [f'Pre-{y}' for y in anchors]:
    row_sub = pivot[pivot['Anchor'] == anchor]
    if len(row_sub) == 0: continue
    row = f"| **{anchor}** | "
    r = []
    for test_year in eval_years:
        if test_year < int(anchor[-4:]): r.append("—")
        else:
            win_fam = winners.loc[(anchor, test_year)] if (anchor, test_year) in winners.index else None
            if win_fam is None:
                r.append("—")
            else:
                max_val = pivot_mat.loc[win_fam, (anchor, test_year)]
                s = f"{max_val:.3f} ({win_fam})"
                if win_fam == 'Linear':
                    s = f"**{s}**"
                r.append(s)
    tex_lines.append(row + " | ".join(r) + " |")

print('\n'.join(tex_lines))
