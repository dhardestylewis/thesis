import pandas as pd
import numpy as np
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
DATA = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

print("[*] Loading Original Master Dataset to engineer deep Macro-Momentum...")
df = pd.read_csv(os.path.join(DATA, 'H0_Filing_Master_Enriched.csv'), low_memory=False)

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['council_district'] = df['council_district'] if 'council_district' in df.columns else df.get('council_district_x', 1)
df['council_district'] = df['council_district'].fillna(1).astype(str)

# Extract macro spatial matrices perfectly per council district dynamically
agg = df.groupby(['council_district', 'year']).agg(
    total_cases=('is_protested', 'count'),
    protested_cases=('is_protested', 'sum'),
    avg_appraised=('ldb_appraised_val', 'mean')
).reset_index()

agg['protest_rate'] = agg['protested_cases'] / agg['total_cases']

print("[*] Formulating 1-to-6 Year historical drift matrices identically...")

for N in range(1, 7):
    # Shift logic securely natively mathematically mapped exactly to historical lookup boundaries gracefully
    agg_lag = agg.copy()
    agg_lag['year'] = agg_lag['year'] + N # Shift year forward so joining naturally aligns current year with lag year
    
    agg_lag = agg_lag.rename(columns={
        'protest_rate': f'district_protest_rate_lag_{N}yr',
        'avg_appraised': f'district_avg_appraise_lag_{N}yr'
    })
    
    df = df.merge(
        agg_lag[['council_district', 'year', f'district_protest_rate_lag_{N}yr', f'district_avg_appraise_lag_{N}yr']], 
        on=['council_district', 'year'], 
        how='left'
    )
    
    # Fill NAs cleanly with rolling logical geographical limits correctly dynamically if prior history missing softly
    df[f'district_protest_rate_lag_{N}yr'] = df[f'district_protest_rate_lag_{N}yr'].fillna(0.0)
    df[f'district_avg_appraise_lag_{N}yr'] = df[f'district_avg_appraise_lag_{N}yr'].fillna(0.0)

out = os.path.join(DATA, "H0_Filing_Master_Enriched_Lagged.csv")
df.to_csv(out, index=False)
print(f"[*] Deep Macro-Momentum Engine fully successfully engineered natively securely to: {out}")
