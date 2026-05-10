import pandas as pd

# Check stage_c column names and key fields
sc = pd.read_parquet('Data/interim/stage_c_features_raw.parquet')
print('stage_c columns:')
print(sc.columns.tolist())
print()

# ID columns
id_cols = [c for c in sc.columns if any(x in c.lower() for x in ['account','tcad','parcel','pid'])]
print('ID columns found:', id_cols)
for c in id_cols:
    vals = sc[c].dropna().head(3).values
    print(f'  {c}: {vals}')

print()
ears = pd.read_csv('Data/Panel/Intermediate/ears_2022_clean.csv', low_memory=False,
                   usecols=['account_number', '_parcel_id_10'])
print(f'EARS account_number sample: {ears["account_number"].dropna().head(5).values}')
print(f'EARS _parcel_id_10 sample: {ears["_parcel_id_10"].dropna().head(5).values}')

print()
enr = pd.read_csv('Data/Warehouse_As_Of/canonical/canonical/H0_Filing_Master_Enriched_v2.csv', low_memory=False)
id_cols_enr = [c for c in enr.columns if any(x in c.lower() for x in ['tcad','account','parcel','pid'])]
print('H0 Enriched ID columns:', id_cols_enr)
for c in id_cols_enr:
    pct = enr[c].notna().mean()*100
    val = enr[c].dropna().head(2).values
    print(f'  {c}: {pct:.1f}% populated, sample={val}')

# try crosswalk
xw = pd.read_csv('Data/Panel/Reference/id_crosswalk.csv', low_memory=False)
print()
print('id_crosswalk sample:')
print(xw.head(3).to_string())
