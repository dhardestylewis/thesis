import os
import pandas as pd
import numpy as np

# Paths
BASE = r"C:\Users\dhl\data\Thesis\thesis\Data\Panel"
CROSSWALK_CSV = r"c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\zoning_land_use_merged_data.csv"
PANEL_CSV = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
LDB_2021 = r"C:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\LDB_2021_kk8y-6cmt.csv"
LDB_2016 = r"C:\Users\dhl\data\Thesis\thesis\Data\CoA_Open_Data\LDB_2016_4nsn-uea6.csv"
OUT_CSV = os.path.join(BASE, "temporal_case_heights.csv")

print("1. Loading Crosswalk and Active Cases...")
cw = pd.read_csv(CROSSWALK_CSV, usecols=["case_number", "tcad_id"])
cw["join_tcad_id"] = cw["tcad_id"].astype(str).str.split('.').str[0].str.zfill(10)
# Map one representative parcel per case for baseline
cw = cw.drop_duplicates(subset=["case_number"])

panel = pd.read_csv(PANEL_CSV, low_memory=False)
cases = panel[panel["period_seq"] == 1][["case_number", "year", "existing_max_height_ft"]].copy()

# Join TCAD ID
cases = cases.merge(cw, on="case_number", how="left")

print("2. Loading LDB Databases...")
ldb_21 = pd.read_csv(LDB_2021, usecols=['PID_10', 'BASEZONE'], on_bad_lines='skip', engine='python')
ldb_21 = ldb_21[ldb_21['PID_10'].notna() & ldb_21['BASEZONE'].notna()].copy()
ldb_21["join_tcad_id"] = ldb_21["PID_10"].astype(str).str.split('.').str[0].str.zfill(10)
ldb_21 = ldb_21.drop_duplicates(subset=["join_tcad_id"])

ldb_16 = pd.read_csv(LDB_2016, usecols=['PID_10', 'BASEZONE'], on_bad_lines='skip', engine='python')
ldb_16 = ldb_16[ldb_16['PID_10'].notna() & ldb_16['BASEZONE'].notna()].copy()
ldb_16["join_tcad_id"] = ldb_16["PID_10"].astype(str).str.split('.').str[0].str.zfill(10)
ldb_16 = ldb_16.drop_duplicates(subset=["join_tcad_id"])

print("3. Executing Temporal Join...")
# Merge both LDBs onto cases
cases = cases.merge(ldb_16[["join_tcad_id", "BASEZONE"]].rename(columns={"BASEZONE": "bz_16"}), on="join_tcad_id", how="left")
cases = cases.merge(ldb_21[["join_tcad_id", "BASEZONE"]].rename(columns={"BASEZONE": "bz_21"}), on="join_tcad_id", how="left")

# Temporal Logic:
# Year <= 2020 -> Use 2016, fallback 2021
# Year >= 2021 -> Use 2021, fallback 2016
def resolve_basezone(row):
    if pd.isna(row["year"]): return np.nan
    if row["year"] <= 2020:
        return row["bz_16"] if pd.notna(row["bz_16"]) else row["bz_21"]
    else:
        return row["bz_21"] if pd.notna(row["bz_21"]) else row["bz_16"]

cases["BASEZONE"] = cases.apply(resolve_basezone, axis=1)

print("4. Mapping Statutory Heights...")
def get_height_limit(zone_string):
    if pd.isna(zone_string):
        return 0
    z = str(zone_string).upper()
    if "CBD" in z or "PUD" in z: return 1000
    if "MF-6" in z: return 90
    if "MF-5" in z: return 60
    if "MF-4" in z: return 60
    if "MF-3" in z: return 40
    if "MF-2" in z: return 40
    if "MF-1" in z: return 40
    if "CS" in z or "GR" in z or "LR" in z: return 60
    if "SF" in z: return 35
    return 35

cases["existing_max_height_ft_statutory"] = cases["BASEZONE"].apply(get_height_limit)

matched = cases["BASEZONE"].notna().mean() * 100
print(f"   Temporal Match Rate for Active Cases: {matched:.2f}%")

cases[["case_number", "existing_max_height_ft_statutory"]].to_csv(OUT_CSV, index=False)
print(f"5. Saved Case Mappings to {OUT_CSV}")
