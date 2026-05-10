import pandas as pd, numpy as np, os
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested']).sort_values('year')

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
X_raw_df = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

# Apply Optimal Target Discretization Filter (Security against Identity Hacked Tuning)
phys_floats = ['ldb_appraised_val', 'land_market_value', 'total_market_value', 'gross_site_area_acres', 'deed_acreage', 'ldb_land_acres', 'ldb_lotsize', 'improvement_sq_ft', 'ldb_imprv_sqft']
to_discretize = [c for c in phys_floats if c in X_raw_df.columns]

if len(to_discretize) > 0:
    for col in to_discretize:
        dt = DecisionTreeClassifier(max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
        X_col = X_raw_df[[col]].values
        dt.fit(X_col, df['is_protested'].values)
        X_raw_df[col] = dt.apply(X_col)

X_raw = X_raw_df.values
y = df['is_protested'].values
years = df['year'].values

anchors_to_test = [2018, 2019, 2020, 2021, 2022]
results = []

for anchor in anchors_to_test:
    train_mask = years < anchor
    if train_mask.sum() == 0: continue

    X_train_raw = X_raw[train_mask]
    y_train = y[train_mask]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)

    # 35 MODEL CONFIGURATIONS
    models = {
        # --- 1. DEFAULT BASELINE ---
        'CatBoost_Default': CatBoostClassifier(iterations=100, depth=6, random_seed=42, verbose=0),
        'XGBoost_Default': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss'),
        'RandomForest_Default': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'LightGBM_Default': LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1),
        'Logistic_Default': LogisticRegression(class_weight='balanced', random_state=42, max_iter=200),
        'TabNet_Default': TabNetClassifier(verbose=0, seed=42),
        'TabNetVREx_Default': TabNetClassifier(optimizer_params={'weight_decay': 0.05}, verbose=0, seed=42),
        
        # --- 2. HIGH-CAPACITY ---
        'CatBoost_HighCap': CatBoostClassifier(iterations=300, depth=10, random_seed=42, verbose=0),
        'XGBoost_HighCap': XGBClassifier(n_estimators=300, max_depth=10, random_state=42, eval_metric='logloss'),
        'RandomForest_HighCap': RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42),
        'LightGBM_HighCap': LGBMClassifier(n_estimators=300, max_depth=15, num_leaves=100, random_state=42, verbose=-1),
        'Logistic_HighCap': LogisticRegression(C=10.0, class_weight='balanced', random_state=42, max_iter=500),
        'TabNet_HighCap': TabNetClassifier(n_d=64, n_a=64, n_steps=7, gamma=1.5, optimizer_params={'weight_decay': 1e-4}, verbose=0, seed=42),
        'TabNetVREx_HighCap': TabNetClassifier(n_d=64, n_a=64, n_steps=7, gamma=1.5, optimizer_params={'weight_decay': 0.005}, verbose=0, seed=42),
        
        # --- 3. HIGHLY-REGULARIZED ---
        'CatBoost_Regularized': CatBoostClassifier(iterations=100, depth=3, l2_leaf_reg=10.0, random_seed=42, verbose=0),
        'XGBoost_Regularized': XGBClassifier(n_estimators=100, max_depth=3, reg_lambda=10.0, reg_alpha=1.0, random_state=42, eval_metric='logloss'),
        'RandomForest_Regularized': RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=50, random_state=42),
        'LightGBM_Regularized': LGBMClassifier(n_estimators=100, max_depth=3, reg_lambda=10.0, random_state=42, verbose=-1),
        'Logistic_Regularized': LogisticRegression(penalty='l2', C=0.01, class_weight='balanced', random_state=42, max_iter=200),
        'TabNet_Regularized': TabNetClassifier(n_d=16, n_steps=3, gamma=1.0, optimizer_params={'weight_decay': 0.1}, verbose=0, seed=42),
        'TabNetVREx_Regularized': TabNetClassifier(n_d=16, n_steps=3, gamma=1.0, optimizer_params={'weight_decay': 0.2}, verbose=0, seed=42),
        
        # --- 4. EXTREME-SHALLOW ---
        'CatBoost_ExtShallow': CatBoostClassifier(iterations=50, depth=2, random_seed=42, verbose=0),
        'XGBoost_ExtShallow': XGBClassifier(n_estimators=50, max_depth=2, random_state=42, eval_metric='logloss'),
        'RandomForest_ExtShallow': RandomForestClassifier(n_estimators=50, max_depth=2, random_state=42),
        'LightGBM_ExtShallow': LGBMClassifier(n_estimators=50, max_depth=2, random_state=42, verbose=-1),
        'Logistic_ExtShallow': LogisticRegression(penalty='l1', C=0.001, solver='liblinear', class_weight='balanced', random_state=42, max_iter=200),
        'TabNet_ExtShallow': TabNetClassifier(n_d=8, n_steps=1, optimizer_params={'weight_decay': 0.2}, verbose=0, seed=42),
        'TabNetVREx_ExtShallow': TabNetClassifier(n_d=8, n_steps=1, optimizer_params={'weight_decay': 0.3}, verbose=0, seed=42),
        
        # --- 5. EXTREME-DEEP ---
        'CatBoost_ExtDeep': CatBoostClassifier(iterations=500, depth=14, random_seed=42, verbose=0),
        'XGBoost_ExtDeep': XGBClassifier(n_estimators=500, max_depth=16, random_state=42, eval_metric='logloss', learning_rate=0.01),
        'RandomForest_ExtDeep': RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42),
        'LightGBM_ExtDeep': LGBMClassifier(n_estimators=500, max_depth=-1, num_leaves=256, random_state=42, verbose=-1),
        'Logistic_ExtDeep': LogisticRegression(C=100.0, class_weight='balanced', random_state=42, max_iter=1000),
        'TabNet_ExtDeep': TabNetClassifier(n_d=128, n_a=128, n_steps=10, gamma=2.0, optimizer_params={'weight_decay': 1e-5}, verbose=0, seed=42),
        'TabNetVREx_ExtDeep': TabNetClassifier(n_d=128, n_a=128, n_steps=10, gamma=2.0, optimizer_params={'weight_decay': 0.001}, verbose=0, seed=42)
    }

    print(f"\n[*] Training 35 Configurations on Pre-{anchor} Anchor...")
    fitted = {}
    for name, m in models.items():
        # ExtDeep TabNet might OOM if epochs are too high, but let's constrain training limits slightly
        if 'TabNet' in name:
            epochs = 5 if 'ExtShallow' in name else (35 if 'ExtDeep' in name else (25 if 'HighCap' in name else 15))
            m.fit(X_train_sc, y_train, max_epochs=epochs, patience=5)
        elif 'Logistic' in name:
            m.fit(X_train_sc, y_train)
        else:
            m.fit(X_train_raw, y_train)
        fitted[name] = m

    eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    print(f"[*] Evaluating OOS Longitudinal Drift for Pre-{anchor} Models...")
    for test_year in eval_years:
        if test_year <= anchor: continue # Only evaluate out-of-sample forward drift
            
        test_mask = years == test_year
        if test_mask.sum() == 0: continue
            
        X_test_raw, y_test = X_raw[test_mask], y[test_mask]
        X_test_sc = scaler.transform(X_test_raw)
        
        for name, m in fitted.items():
            if 'TabNet' in name or 'Logistic' in name:
                p = m.predict_proba(X_test_sc)[:, 1]
            else:
                p = m.predict_proba(X_test_raw)[:, 1]
                
            prauc = average_precision_score(y_test, p)
            
            results.append({
                'Anchor': f'Pre-{anchor}',
                'Model': name.split('_')[0],
                'Profile': name.split('_')[1],
                'Evaluate_Year': test_year,
                'PRAUC': prauc
            })

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(ROOT, 'grid_tuning_results_expanded.csv'), index=False)
print("[*] Completed massive 175-model expanded parameter sweep. Saved to grid_tuning_results_expanded.csv.")

