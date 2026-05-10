import pandas as pd, numpy as np, os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)

phys_floats = [
    'ldb_appraised_val', 'land_market_value', 'total_market_value',
    'gross_site_area_acres', 'deed_acreage', 'ldb_land_acres', 'ldb_lotsize',
    'improvement_sq_ft', 'ldb_imprv_sqft'
]

targets = [c for c in phys_floats if c in df.columns]
n = len(df)

results = []
for col in targets:
    series = df[col].dropna()
    q75, q25 = np.percentile(series, [75 ,25])
    iqr = q75 - q25
    if iqr == 0:
        k = 10 # fallback
        rule = "IQR is 0 (Fallback)"
    else:
        bin_width = 2 * iqr * (n ** (-1/3))
        v_min, v_max = series.min(), series.max()
        k = int(np.ceil((v_max - v_min) / bin_width))
        rule = "Freedman-Diaconis"
    
    # Sturges
    sturges_k = int(np.ceil(np.log2(n) + 1))
    
    results.append({
        'Feature': col,
        'Freedman-Diaconis (k)': min(k, 100), # Cap extreme values
        'Uncapped FD': k,
        'Sturges (k)': sturges_k
    })
    
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

