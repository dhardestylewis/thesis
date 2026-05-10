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
        'CatBoost': CatBoostClassifier(iterations=100, depth=6, verbose=0, random_seed=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Logistic (L2)': LogisticRegression(class_weight='balanced', random_state=42)
    }

    for name, m in models.items():
        if name == 'Logistic (L2)': m.fit(X_train_sc, y_train)
        else: m.fit(X_train_raw, y_train)

    for test_year in eval_years:
        if test_year < anchor: continue
        test_mask = years == test_year
        if test_mask.sum() < 5 or y[test_mask].sum() < 1: continue
            
        X_test_raw, y_test = X_raw.values[test_mask], y[test_mask]
        X_test_sc = scaler.transform(X_test_raw)
        
        for name, m in models.items():
            p = m.predict_proba(X_test_sc if name == 'Logistic (L2)' else X_test_raw)[:, 1]
            prauc = average_precision_score(y_test, p)
            drift_results.append({
                'Model': name, 'Anchor': f'{anchor}',
                'Evaluate_Year': test_year, 'Offset': test_year - anchor, 'PRAUC': prauc
            })

results_df = pd.DataFrame(drift_results)

# Find winners
print("\n--- WINNER TALLY (OVERALL) ---")
pivot = results_df.pivot_table(index='Model', columns=['Anchor', 'Evaluate_Year'], values='PRAUC')
winners = pivot.idxmax()
print(winners.value_counts())

print("\n--- WINNER BY HORIZON ---")
for offset in sorted(results_df['Offset'].unique()):
    sub = results_df[results_df['Offset'] == offset]
    pw = sub.groupby(['Anchor', 'Evaluate_Year']).apply(lambda x: x.loc[x['PRAUC'].idxmax(), 'Model']).value_counts()
    print(f"\nHorizon T+{offset}:")
    print(pw)

print("\n--- WINNER BY ANCHOR PERIOD ---")
for anchor in results_df['Anchor'].unique():
    sub = results_df[results_df['Anchor'] == anchor]
    pw = sub.groupby('Evaluate_Year').apply(lambda x: x.loc[x['PRAUC'].idxmax(), 'Model']).value_counts()
    print(f"\nTrained on Pre-{anchor}:")
    print(pw)
