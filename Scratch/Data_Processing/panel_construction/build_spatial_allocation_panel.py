import os
import pandas as pd
import numpy as np

BASE = r"c:\Users\dhl\data\Thesis\thesis\Data\Panel"
UNIVERSE_CSV = os.path.join(BASE, "parcel", "property_universe.csv")
ACS_CSV = os.path.join(BASE, "census", "acs_tract_timeseries.csv")
OUT_CSV = os.path.join(BASE, "spatial_allocation_panel.csv")

print("1. Loading Property Universe...")
df = pd.read_csv(UNIVERSE_CSV, low_memory=False)
print(f"   Loaded {len(df)} parcels from Travis County.")

print("\n2. Defining the Target (is_rezoned)...")
# A parcel is considered "rezoned" or targeted for discretionary zoning if it has a zoning_case_GEOID attached.
df["is_rezoned"] = df["zoning_case_GEOID"].notna().astype(int)
print(f"   Target Breakdown: {df['is_rezoned'].value_counts().to_dict()}")

print("\n3. Hydrating Demographics (ACS 2022 Static Snapshot)...")
acs = pd.read_csv(ACS_CSV)
# Filter ACS to 2022 to act as the static demographic state for our cross-sectional model
acs = acs.rename(columns={"vintage": "year"})
acs_2022 = acs[acs["year"] == 2022].copy()
acs_2022["geoid_tract"] = acs_2022["geoid_tract"].astype(str)

# Engineer Ratios
acs_2022["renter_share"] = acs_2022["renter_occupied_units"] / acs_2022["total_housing_units"]
acs_2022["rent_burden"] = acs_2022["median_gross_rent"] / (acs_2022["median_household_income"]/12)
acs_2022["affordability_proxy"] = acs_2022["median_household_income"] / acs_2022["median_home_value"]

# Clean nearby_GEOID on the parcel side
df["nearby_GEOID"] = df["nearby_GEOID"].astype(str).str.split('.').str[0]

# Merge Demographics
df = df.merge(acs_2022, left_on="nearby_GEOID", right_on="geoid_tract", how="left")

# Basic feature cleaning for the panel
df["lui_shape_area"] = pd.to_numeric(df["lui_shape_area"], errors="coerce")
df["lui_general_land_use"] = df["lui_general_land_use"].astype(str).fillna("Unknown")

# Handle missing ACS data with median imputation for the CatBoost baseline
acs_features = [
    "total_population", "median_household_income", "median_home_value",
    "median_gross_rent", "renter_share", "rent_burden", "affordability_proxy",
    "race_white", "race_black", "race_hispanic", "median_age"
]
for col in acs_features:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

out_cols = ["standardized_tcad_id", "is_rezoned", "lui_general_land_use", "lui_shape_area", "council_district"] + acs_features
df_final = df[out_cols].copy()

df_final.to_csv(OUT_CSV, index=False)
print(f"\n4. Saved Spatial Allocation Panel: {OUT_CSV}")
print(f"   Shape: {df_final.shape}")
