import os, pandas as pd

coa_dir = 'Data/CoA_Open_Data'
panel_dir = 'Data/Panel'

print('=== CoA_Open_Data files ===')
for f in os.listdir(coa_dir):
    path = os.path.join(coa_dir, f)
    if os.path.isfile(path):
        sz = os.path.getsize(path)/1e6
        print(f'  {f}: {sz:.1f} MB')

print()
print('=== Panel files ===')
for f in os.listdir(panel_dir):
    path = os.path.join(panel_dir, f)
    if os.path.isfile(path):
        sz = os.path.getsize(path)/1e6
        print(f'  {f}: {sz:.1f} MB')

pu_path = os.path.join(panel_dir, 'property_universe.csv')
print()
print('property_universe.csv exists:', os.path.exists(pu_path))

ldb_files = {
    2016: os.path.join(coa_dir, 'LDB_2016_4nsn-uea6.csv'),
    2021: os.path.join(coa_dir, 'LDB_2021_kk8y-6cmt.csv'),
}
print()
print('=== LDB files ===')
for year, path in ldb_files.items():
    exists = os.path.exists(path)
    sz = os.path.getsize(path)/1e6 if exists else 0
    print(f'  LDB_{year}: {"EXISTS" if exists else "MISSING"} ({sz:.0f} MB)')

sc = pd.read_parquet('Data/interim/stage_c_features_raw.parquet')
has_latlon = sc['latitude'].notna() & sc['longitude'].notna()
print()
print(f'Cases with lat/lon: {has_latlon.sum()} / {len(sc)} ({has_latlon.mean()*100:.1f}%)')

for g in ['nearby_GEOID', 'zoning_case_GEOID']:
    if g in sc.columns:
        pct = sc[g].notna().mean()*100
        print(f'{g}: {pct:.1f}% populated')

print()
print('=== standardized_tcad_id coverage ===')
enr = pd.read_csv('Data/Warehouse_As_Of/canonical/canonical/H0_Filing_Master_Enriched_v2.csv', low_memory=False)
if 'standardized_tcad_id' in enr.columns:
    pct = enr['standardized_tcad_id'].notna().mean()*100
    print(f'standardized_tcad_id: {pct:.1f}% populated')
else:
    print('standardized_tcad_id: NOT IN ENRICHED FILE')
    print('Columns with tcad/id:', [c for c in enr.columns if 'tcad' in c.lower() or 'pid' in c.lower()])
