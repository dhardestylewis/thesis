import pandas as pd, numpy as np, os
from sklearn.tree import DecisionTreeClassifier

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)

phys_floats = [
    'ldb_appraised_val', 'land_market_value', 'total_market_value',
    'gross_site_area_acres', 'deed_acreage', 'ldb_land_acres', 'ldb_lotsize',
    'improvement_sq_ft', 'ldb_imprv_sqft'
]

targets = [c for c in phys_floats if c in df.columns]
df['is_protested'] = pd.to_numeric(df['is_protested'], errors='coerce')
df = df.dropna(subset=['is_protested'])
y = df['is_protested'].values

results = []
print("======= NEW RELAXED ENTROPY BOUNDARIES =======")
for col in targets:
    series = pd.to_numeric(df[col], errors='coerce').fillna(0).values.reshape(-1, 1)
    
    # Relaxed constraints allowing up to 20 highly granular bins as long as 30 properties exist per bin
    dt = DecisionTreeClassifier(max_depth=None, max_leaf_nodes=20, min_samples_leaf=30, random_state=42)
    dt.fit(series, y)
    
    n_bins = dt.tree_.n_leaves
    print(f"{col}: {n_bins} leaves")
        
