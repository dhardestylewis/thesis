import pandas as pd
import os

print("=== 1. PETITION COLUMNS ===")
pet = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\petition_signers_backfilled.csv')
pet['date_parsed'] = pd.to_datetime(pet['date'], errors='coerce')
print("Columns:", pet.columns.tolist())
print("Rows:", len(pet))
print("date sample:", pet['date'].head(5).tolist())
print("date parsed non-null:", pet['date_parsed'].notna().sum())
print("signed dtype:", pet['signed'].dtype, "| values:", pet['signed'].value_counts().head().to_dict())
print("area_pct sample:", pet['area_pct'].describe().to_dict())
print()

print("=== 2. PARCEL CROSSWALK COVERAGE ===")
zoning = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv',
                     low_memory=False, usecols=['case_number','parcel_id_10'])
crosswalk = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\Panel\Reference\id_crosswalk.csv')
crosswalk['parcel_id_10'] = crosswalk['parcel_id_10'].astype(str).str.zfill(10)
zoning['parcel_id_10_str'] = zoning['parcel_id_10'].apply(
    lambda x: str(int(float(x))).zfill(10) if pd.notna(x) else None)
zoning_xw = zoning.merge(crosswalk, left_on='parcel_id_10_str', right_on='parcel_id_10', how='left')
print(f"Zoning cases total: {len(zoning)}")
print(f"Have parcel_id_10: {zoning['parcel_id_10'].notna().sum()}")
print(f"Crosswalk matched (ears_account_number): {zoning_xw['ears_account_number'].notna().sum()}")
print(f"Cases with no parcel link at all: {zoning['parcel_id_10'].isna().sum()}")
print()

print("=== 3. ENRICHED FILE COVERAGE ===")
enr = pd.read_csv(
    r'c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\enriched_zoning_data_causal.csv',
    low_memory=False, usecols=['case_number','proposed_max_height_ft'])
print(f"Enriched rows: {len(enr)}, unique cases: {enr['case_number'].nunique()}")
print(f"height non-null: {enr['proposed_max_height_ft'].notna().sum()}")
# What script generates this file?
scratch_dir = r'c:\Users\dhl\data\Thesis\thesis\Scratch'
scripts = [f for f in os.listdir(scratch_dir) if 'enrich' in f.lower() or 'causal' in f.lower() or 'metric' in f.lower()]
print("Scripts that may generate enriched file:", scripts)
print()

print("=== 4. REAL_DAYS_IN_PIPELINE NULLS ===")
z2 = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv',
                 low_memory=False, usecols=['case_number','label_real_days_in_pipeline','Final_Council_Date','Derived_Status'])
null_days = z2[z2['label_real_days_in_pipeline'].isna()]
print(f"Cases with null label_real_days_in_pipeline: {len(null_days)}")
print("Derived_Status for nulls:")
print(null_days['Derived_Status'].value_counts().head(10).to_string())
print("Final_Council_Date null count:", null_days['Final_Council_Date'].isna().sum())
