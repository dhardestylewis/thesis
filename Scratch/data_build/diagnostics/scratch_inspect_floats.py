import pandas as pd, numpy as np, os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)

drop_cols = ['is_protested', 'case_number', 'organized_opposition', 'year', 'date', 'application_start_date', 'final_date']
future_features = ['staff_recommendation_cat', 'agenda_text_raw', 'spatial_contagion_1yr', 'spatial_contagion_3yr']
X = df.drop(columns=[c for c in (drop_cols + future_features) if c in df.columns], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

total_rows = len(X)
print(f"Total Rows: {total_rows}")
results = []

for c in X.columns:
    unique_vals = X[c].nunique()
    uniqueness = unique_vals / total_rows
    dtype = X[c].dtype
    results.append({
        'Feature': c,
        'UniqueCount': unique_vals,
        'Uniqueness': uniqueness,
        'DType': dtype
    })

res_df = pd.DataFrame(results).sort_values('Uniqueness', ascending=False)
print("\n--- Top 25 Most Unique Features (High Memorization Risk) ---")
print(res_df.head(25).to_string(index=False))

print("\n--- Bottom 10 Features (Already Binned / Dummies) ---")
print(res_df.tail(10).to_string(index=False))

