import os
import pandas as pd
import numpy as np

BASE = r"C:\Users\dhl\data\Thesis\thesis\Data\Panel"
ALLOCATION_CSV = os.path.join(BASE, "spatial_allocation_panel.csv")
LDB_2021 = r"C:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\LDB_2021_kk8y-6cmt.csv"
LDB_2016 = r"C:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\LDB_2016_4nsn-uea6.csv"

print("1. Loading Datasets...")
panel_df = pd.read_csv(ALLOCATION_CSV, low_memory=False)

# Clean TCAD ID on panel side
panel_df["join_tcad_id"] = panel_df["standardized_tcad_id"].astype(str).str.split('.').str[0].str.zfill(10)

# Drop previously merged columns if they exist
cols_to_drop = ["BASEZONE", "existing_max_height_ft_mapped"]
panel_df = panel_df.drop(columns=[c for c in cols_to_drop if c in panel_df.columns])

# Load LDB 2016 (Only use 2016 to prevent post-treatment leakage from 2021)
ldb_16 = pd.read_csv(LDB_2016, usecols=['PID_10', 'BASEZONE'], on_bad_lines='skip', engine='python')
ldb_16 = ldb_16[ldb_16['PID_10'].notna() & ldb_16['BASEZONE'].notna()].copy()
ldb_16["join_tcad_id"] = ldb_16["PID_10"].astype(str).str.split('.').str[0].str.zfill(10)
ldb_df = ldb_16.drop_duplicates(subset=["join_tcad_id"])

print(f"   Panel size: {len(panel_df)}")
print(f"   LDB size: {len(ldb_df)}")

print("\n2. Merging Base Zoning...")
merged = panel_df.merge(ldb_df[["join_tcad_id", "BASEZONE"]], on="join_tcad_id", how="left")
matched_pct = merged["BASEZONE"].notna().mean() * 100
print(f"   Match Rate: {matched_pct:.2f}%")

print("\n3. Mapping Statutory Heights...")
# Dictionary mapping statutory max heights from Austin Land Development Code (Title 25)
def get_height_limit(zone_string):
    if pd.isna(zone_string):
        return 0 # If outside city limits or missing, default 0
    z = str(zone_string).upper()
    # PUD / CBD
    if "CBD" in z or "PUD" in z: return 1000 # Uncapped or negotiated
    # Multi-family and Commercial
    if "MF-6" in z: return 90
    if "MF-5" in z: return 60
    if "MF-4" in z: return 60
    if "MF-3" in z: return 40
    if "MF-2" in z: return 40
    if "MF-1" in z: return 40
    if "CS" in z or "GR" in z or "LR" in z: return 60
    # Single Family
    if "SF" in z: return 35
    # Default catch-all for other residential / unknown
    return 35

merged["existing_max_height_ft_mapped"] = merged["BASEZONE"].apply(get_height_limit)

# Ensure the column exists for the Universal Model
merged.to_csv(ALLOCATION_CSV, index=False)
print(f"\n4. Saved Updated Panel: {ALLOCATION_CSV}")
print("   Ready for Universal Model Rerun.")
