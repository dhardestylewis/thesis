import pandas as pd

ldb21 = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\LDB_2021_kk8y-6cmt.csv',
    low_memory=False, usecols=['PID_10','PROP_ID','FAR','BASEZONE'])
master = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\Final\model_ready_spatial_hydrated.csv',
    low_memory=False)

ldb21['PID_10_norm'] = ldb21['PID_10'].astype(str).str.zfill(10)
ldb21_dedup = ldb21.sort_values('FAR', ascending=False).drop_duplicates('PID_10_norm', keep='first')

master['PID_10_norm'] = master['parcel_id_10'].apply(
    lambda x: str(int(x)).zfill(10) if pd.notna(x) else None)

matched_mask = master['PID_10_norm'].notna() & master['PID_10_norm'].isin(ldb21_dedup['PID_10_norm'])
unmatched = master[~matched_mask].copy()
matched = master[matched_mask].copy()

print(f"Total cases: {len(master)}")
print(f"Matched via PID_10: {len(matched)}")
print(f"Unmatched: {len(unmatched)}")
print()

no_parcel = master['parcel_id_10'].isna().sum()
has_parcel_no_ldb = (master['PID_10_norm'].notna() & ~master['PID_10_norm'].isin(ldb21_dedup['PID_10_norm'])).sum()
print(f"  Reason 1 - No parcel_id_10 in master: {no_parcel}")
print(f"  Reason 2 - Has parcel_id_10 but not in LDB 2021 (ETJ/non-COA): {has_parcel_no_ldb}")

# Spatial fallback potential
has_ll = unmatched['latitude'].notna() & unmatched['longitude'].notna()
print(f"\nUnmatched with lat/lon (spatial join possible): {has_ll.sum()} / {len(unmatched)}")

# TCAD ID analysis for no-parcel cases
no_parcel_df = master[master['parcel_id_10'].isna()]
print(f"\nNo-parcel cases with lat/lon: {(no_parcel_df['latitude'].notna() & no_parcel_df['longitude'].notna()).sum()} / {len(no_parcel_df)}")
print("\ntcad_id value counts for no-parcel cases (top 10):")
print(no_parcel_df['tcad_id'].value_counts().head(10))

# Check property_id column
print("\nproperty_id coverage in master:")
print(master['property_id'].notna().sum(), "non-null of", len(master))
print("Sample property_id:", master['property_id'].dropna().head(5).tolist())

# Check LDB PROP_ID range vs master property_id
print("\nLDB PROP_ID sample:", ldb21['PROP_ID'].dropna().head(5).tolist())
ldb_propids = set(ldb21['PROP_ID'].dropna().astype(str))
master_propids = set(master['property_id'].dropna().astype(str))
print(f"property_id overlap with LDB PROP_ID: {len(ldb_propids & master_propids)}")
