"""
Track 3: Propensity Score Matching (PSM) for Non-Zoning Controls
1. Identify real protested cases and their subject parcels.
2. Find 1:1 "pseudo-cases" (non-zoning parcels) in the exact same Census GEOID,
   with the exact same general land use, minimizing lot size difference.
3. Run a 200ft spatial join for these control pseudo-cases.
4. Compare baseline neighborhood structures.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import os

BASE    = r"C:\Users\dhl\data\Thesis\thesis\Data"
OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

MASTER_PATH    = os.path.join(BASE, "final", "model_ready_zoning_data.csv")
CENTROIDS_PATH = os.path.join(BASE, "Panel", "Reference", "parcel_centroids.csv")
REC_PATH       = os.path.join(OUT_DIR, "recovered_petitions.csv")
PROP_PATH      = os.path.join(BASE, "Panel", "parcel", "property_universe.csv")
ADJ_PATH       = os.path.join(OUT_DIR, "adjacent_parcel_universe.csv")

print("1. Loading Property Universe...")
pu = pd.read_csv(PROP_PATH, low_memory=False)
pu["parcel_str"] = pu["standardized_tcad_id"].astype(str).str.strip()

print("2. Loading Real Zoning Cases...")
master = pd.read_csv(MASTER_PATH, low_memory=False)
cases = master[["case_number", "latitude", "longitude", "council_district"]].drop_duplicates("case_number").dropna(subset=["latitude", "longitude"])
rec = pd.read_csv(REC_PATH)
protested_cases = set(rec["case_number"].unique())
cases["is_protested"] = cases["case_number"].isin(protested_cases).astype(int)

# We need the GEOID and land use for the real cases to match exactly.
# We will do a quick 1-nearest-neighbor spatial join of real cases to the property universe to find their subject parcel.
print("3. Finding Subject Parcels for Real Cases to extract their GEOID and Land Use...")
cen = pd.read_csv(CENTROIDS_PATH, low_memory=False).dropna(subset=["latitude", "longitude"])
cen["parcel_str"] = cen["parcel_id_10"].astype(str).str.strip()
cen = cen.merge(pu[["parcel_str", "lui_general_land_use", "nearby_GEOID", "lui_shape_area"]], on="parcel_str", how="inner")

tree_cen = BallTree(np.radians(cen[["latitude", "longitude"]].values), metric="haversine")
dist, idx = tree_cen.query(np.radians(cases[["latitude", "longitude"]].values), k=1)

cases["subject_parcel"] = cen.iloc[idx.flatten()]["parcel_str"].values
cases["subject_land_use"] = cen.iloc[idx.flatten()]["lui_general_land_use"].values
cases["subject_geoid"] = cen.iloc[idx.flatten()]["nearby_GEOID"].values
cases["subject_area"] = cen.iloc[idx.flatten()]["lui_shape_area"].values

print("4. Executing 1:1 Matching within Exact GEOID...")
# We want to match protested cases (is_protested == 1) to non-zoning parcels.
# We exclude any parcel that is already a subject parcel of ANY zoning case.
exclude_parcels = set(cases["subject_parcel"])
candidate_pool = cen[~cen["parcel_str"].isin(exclude_parcels)].copy()

protested_df = cases[cases["is_protested"] == 1].copy()
matches = []

for _, row in protested_df.iterrows():
    geoid = row["subject_geoid"]
    land_use = row["subject_land_use"]
    target_area = row["subject_area"]
    
    # Filter candidates to exact same Census GEOID and exact same Land Use
    valid_candidates = candidate_pool[
        (candidate_pool["nearby_GEOID"] == geoid) & 
        (candidate_pool["lui_general_land_use"] == land_use)
    ]
    
    if len(valid_candidates) == 0:
        # Fallback: same council district and land use
        valid_candidates = candidate_pool[
            (candidate_pool["lui_general_land_use"] == land_use)
        ] # simplified fallback
        
    if len(valid_candidates) > 0:
        # Nearest neighbor in lot size
        best_idx = (valid_candidates["lui_shape_area"] - target_area).abs().idxmin()
        best_match = valid_candidates.loc[best_idx]
        matches.append({
            "real_case_number": row["case_number"],
            "pseudo_case_id": "CONTROL_" + row["case_number"],
            "control_parcel_id": best_match["parcel_str"],
            "control_lat": best_match["latitude"],
            "control_lon": best_match["longitude"],
        })
        # Remove from pool to prevent replacement
        candidate_pool = candidate_pool.drop(best_idx)

control_cases = pd.DataFrame(matches)
print(f"   Matched {len(control_cases)} control cases out of {len(protested_df)} protested cases.")

print("5. Running 200ft Spatial Join for Control Cases...")
FT_200 = 200 / 3280.84   # 200 feet in km
EARTH_R = 6371.0         # km

control_coords_rad = np.radians(control_cases[["control_lat", "control_lon"]].values)
indices_200 = tree_cen.query_radius(control_coords_rad, r=FT_200 / EARTH_R)

control_neighbors = []
for i, (ctrl_row, idx_200) in enumerate(zip(control_cases.itertuples(), indices_200)):
    for j in idx_200:
        neighbor = cen.iloc[j]
        control_neighbors.append({
            "pseudo_case_id": ctrl_row.pseudo_case_id,
            "neighbor_parcel": neighbor["parcel_str"],
            "lui_general_land_use": neighbor["lui_general_land_use"],
            "lui_shape_area": neighbor["lui_shape_area"]
        })

cn_df = pd.DataFrame(control_neighbors)
print(f"   Found {len(cn_df)} adjacent parcels for the control pseudo-cases.")

print("6. Comparing Neighborhood Baselines...")
cn_df["is_single_family"] = (cn_df["lui_general_land_use"] == 100).astype(int)
cn_df["is_commercial"] = (cn_df["lui_general_land_use"] == 300).astype(int)

ctrl_summary = pd.DataFrame([{
    "pct_single_family": cn_df["is_single_family"].mean(),
    "pct_commercial": cn_df["is_commercial"].mean(),
    "median_lot_size_sqft": cn_df["lui_shape_area"].median()
}], index=["PSM Controls (Non-Zoning)"])

print("\n=== PSM MATCHED CONTROL NEIGHBORHOODS (0-200ft) ===")
print("These are parcels that look identical to protested cases but did NOT file a zoning case.")
print(ctrl_summary.round(3).to_string())

# Get the real protested cases summary from adjacent_parcel_universe for comparison
adj = pd.read_csv(ADJ_PATH, low_memory=False)
adj["is_single_family"] = (adj["lui_general_land_use"] == 100).astype(int)
adj["is_commercial"] = (adj["lui_general_land_use"] == 300).astype(int)
real_protested = adj[(adj["case_protested"] == 1) & (adj["zone"] == "A_eligible")]

real_summary = pd.DataFrame([{
    "pct_single_family": real_protested["is_single_family"].mean(),
    "pct_commercial": real_protested["is_commercial"].mean(),
    "median_lot_size_sqft": real_protested["lui_shape_area"].median()
}], index=["Real Protested Cases"])

print("\n=== REAL PROTESTED NEIGHBORHOODS (0-200ft) ===")
print(real_summary.round(3).to_string())

# Save artifacts
cn_df.to_csv(rf"{OUT_DIR}\psm_control_neighbors.csv", index=False)
control_cases.to_csv(rf"{OUT_DIR}\psm_control_cases.csv", index=False)
