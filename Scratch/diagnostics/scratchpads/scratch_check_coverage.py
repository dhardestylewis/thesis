import pandas as pd, os

pu = pd.read_csv('Data/Panel/parcel/property_universe.csv', low_memory=False)
print(f'property_universe: {len(pu)} rows')
print(f'Columns: {pu.columns.tolist()}')
print(f'nearby_GEOID populated: {pu["nearby_GEOID"].notna().sum()} ({pu["nearby_GEOID"].notna().mean()*100:.1f}%)')
print(f'zoning_case_GEOID populated: {pu["zoning_case_GEOID"].notna().sum()} ({pu["zoning_case_GEOID"].notna().mean()*100:.1f}%)')
print(f'latitude populated: {pu["latitude"].notna().sum()} ({pu["latitude"].notna().mean()*100:.1f}%)')
print(f'standardized_tcad_id populated: {pu["standardized_tcad_id"].notna().sum()} ({pu["standardized_tcad_id"].notna().mean()*100:.1f}%)')
print()

ears_dir = 'Data/Panel/Intermediate'
ears_files = sorted([f for f in os.listdir(ears_dir) if f.startswith('ears_')])
print(f'EARS files available: {ears_files}')
for f in ears_files:
    sample = pd.read_csv(os.path.join(ears_dir, f), nrows=2, low_memory=False)
    yr = f.replace('ears_','').replace('_clean.csv','')
    pid_col = '_parcel_id_10' if '_parcel_id_10' in sample.columns else 'PID_10'
    acct_col = 'account_number'
    print(f'  {f}: {len(sample.columns)} cols, pid_col={pid_col}, acct sample={sample[acct_col].values[:2] if acct_col in sample.columns else "N/A"}')
