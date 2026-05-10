import pandas as pd, numpy as np, os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, median_absolute_error
from sklearn.linear_model import ElasticNet
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

def median_absolute_percentage_error(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0: return np.nan
    return np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'is_protested', 'case_number']).sort_values('year')

try:
    pet = pd.read_csv(os.path.join(ROOT, "Data", "Protest_Petitions", "Backfilled", "petition_summary_backfilled.csv"))
    df = df.merge(pet[['case_number', 'signer_pct']], on='case_number', how='left')
    df['signed_area_share'] = (df['signer_pct'] / 100.0).fillna(0.0)
except Exception as e:
    raise RuntimeError("Missing Petition file for continuous analysis.")

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date', 'council_district', 'council_district_x', 'signer_pct', 'signed_area_share']
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

X_raw = X_raw_df.values
years = df['year'].values
anchor = 2020
train_mask = years < anchor

y_reg = df['signed_area_share'].values

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)

print(f"[*] Executing Non-Tree Continuous Regression on Pre-{anchor} Anchor...")

reg_models = {
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
    'TabNet_Regressor': TabNetRegressor(verbose=0, seed=42)
}

X_train_sc = X_sc[train_mask]
y_train_reg = y_reg[train_mask]

fitted_regs = {}
for name, m in reg_models.items():
    if 'TabNet' in name:
        m.fit(X_train_sc, y_train_reg.reshape(-1, 1), max_epochs=20, patience=5)
    else:
        m.fit(X_train_sc, y_train_reg)
    fitted_regs[name] = m

eval_years = [2021, 2022, 2023, 2024]
results = []
for test_year in eval_years:
    test_mask = years == test_year
    if test_mask.sum() == 0: continue
    X_test_sc, y_test = X_sc[test_mask], y_reg[test_mask]
    
    for name, m in fitted_regs.items():
        preds = m.predict(X_test_sc)
        if 'TabNet' in name: preds = preds.flatten()
        preds = np.clip(preds, 0, 1) # Bounds
        
        rmse = root_mean_squared_error(y_test, preds)
        mdae = median_absolute_error(y_test, preds)
        mdape = median_absolute_percentage_error(y_test, preds)
        
        results.append({
            'Model': name,
            'Evaluate_Year': test_year,
            'RMSE': round(rmse, 3),
            'MdAE': round(mdae, 4),
            'MdAPE_nonzero': round(mdape, 3) if not np.isnan(mdape) else None
        })

res_df = pd.DataFrame(results)
print("\n=== Non-Tree Zero-Inflated Regression Errors ===")
print(res_df.pivot_table(index='Model', columns='Evaluate_Year', values=['RMSE', 'MdAE']))

print("\n=== Median Absolute Percentage Error (MdAPE) on strictly Non-Zero targets ===")
print(res_df.pivot_table(index='Model', columns='Evaluate_Year', values='MdAPE_nonzero'))

