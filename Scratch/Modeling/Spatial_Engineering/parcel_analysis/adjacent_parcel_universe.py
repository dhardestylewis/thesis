"""
Track 3: Adjacent Parcel Universe — Spatial Join
For every zoning case (all 6,583), find all parcel centroids within:
  Zone A: 0-200 ft  (~61m) — legally eligible to sign protest petition
  Zone B: 200-500 ft (~152m) — affected but legally ineligible
Cross-reference Zone A with recovered_petitions to identify:
  - signers (signed=1 in recovered_petitions)
  - eligible non-signers (in Zone A but not in recovered_petitions for that case)
Output: one row per (case_number x parcel_id) with zone, signed flag, parcel attributes
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import os

BASE    = r"C:\Users\dhl\data\Thesis\thesis\Data"
OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

MASTER_PATH   = os.path.join(BASE, "final", "model_ready_zoning_data.csv")
CENTROIDS_PATH= os.path.join(BASE, "Panel", "Reference", "parcel_centroids.csv")
REC_PATH      = os.path.join(OUT_DIR, "recovered_petitions.csv")
PROP_PATH     = os.path.join(BASE, "Panel", "parcel", "property_universe.csv")

FT_200 = 200 / 3280.84   # 200 feet in km (for haversine)
FT_500 = 500 / 3280.84   # 500 feet in km
EARTH_R= 6371.0           # km

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading master cases...")
master = pd.read_csv(MASTER_PATH, low_memory=False)
# One row per case, with lat/lon of subject parcel
cases = (master[["case_number","latitude","longitude","council_district",
                  "case_type","application_start_date"]]
         .dropna(subset=["latitude","longitude"])
         .drop_duplicates("case_number")
         .copy())
print(f"  Cases with lat/lon: {len(cases):,}")

print("Loading parcel centroids...")
centroids = pd.read_csv(CENTROIDS_PATH, low_memory=False)
centroids = centroids.dropna(subset=["latitude","longitude"]).copy()
# Rename to avoid collision
centroids = centroids.rename(columns={
    "latitude": "p_lat", "longitude": "p_lon",
    "parcel_id_10": "parcel_id"
})
print(f"  Parcels loaded: {len(centroids):,}")

print("Loading recovered petitions...")
rec = pd.read_csv(REC_PATH)
rec["tcad_str"] = rec["tcad_id"].astype(str).str.strip()
centroids["parcel_str"] = centroids["parcel_id"].astype(str).str.strip()

print("Loading property universe for land-use attributes...")
pu = pd.read_csv(PROP_PATH, low_memory=False)
pu["parcel_str"] = pu["standardized_tcad_id"].astype(str).str.strip()
pu = pu[["parcel_str","lui_land_use","lui_general_land_use","lui_shape_area"]].drop_duplicates("parcel_str")

# ── Build BallTree on all parcel centroids (radians for haversine) ───────────
print("Building BallTree...")
coords_rad = np.radians(centroids[["p_lat","p_lon"]].values)
tree = BallTree(coords_rad, metric="haversine")
print(f"  BallTree built on {len(centroids):,} parcels")

# ── Spatial join: query each case for neighbors in 0-500 ft ─────────────────
print("Running spatial join (all cases, 0-500 ft)...")
case_coords_rad = np.radians(cases[["latitude","longitude"]].values)

# Query at 500 ft radius
indices_500 = tree.query_radius(case_coords_rad, r=FT_500 / EARTH_R)
# Query at 200 ft radius
indices_200 = tree.query_radius(case_coords_rad, r=FT_200 / EARTH_R)

print("Assembling output...")
rows = []
for i, (case_row, idx_500, idx_200) in enumerate(
        zip(cases.itertuples(), indices_500, indices_200)):
    if i % 500 == 0:
        print(f"  {i:,}/{len(cases):,} cases processed...", flush=True)
    
    idx_200_set = set(idx_200)
    case_num = case_row.case_number
    
    # Signers for this case from recovered petitions
    signers = set(
        rec[rec["case_number"] == case_num]["tcad_str"].values
    )
    
    for j in idx_500:
        parcel = centroids.iloc[j]
        p_id   = parcel["parcel_str"]
        in_200 = j in idx_200_set
        
        zone = "A_eligible" if in_200 else "B_ineligible"
        signed = 1 if (in_200 and p_id in signers) else 0
        
        rows.append({
            "case_number":      case_num,
            "parcel_id":        p_id,
            "zone":             zone,
            "signed":           signed,
            "p_lat":            parcel["p_lat"],
            "p_lon":            parcel["p_lon"],
            "case_council_district": case_row.council_district,
            "case_type":        case_row.case_type,
        })

result = pd.DataFrame(rows)
print(f"\nTotal parcel-case pairs: {len(result):,}")

# ── Merge property attributes ────────────────────────────────────────────────
result = result.merge(pu, left_on="parcel_id", right_on="parcel_str", how="left")

# ── Flag whether each case ever had a petition ───────────────────────────────
protested_cases = set(rec["case_number"].unique())
result["case_protested"] = result["case_number"].isin(protested_cases).astype(int)

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Unique cases covered:     {result['case_number'].nunique():,}")
print(f"Unique parcels covered:   {result['parcel_id'].nunique():,}")
print(f"Zone A (eligible, 0-200ft): {(result['zone']=='A_eligible').sum():,}")
print(f"Zone B (ineligible, 200-500ft): {(result['zone']=='B_ineligible').sum():,}")
print(f"Signed parcels (Zone A):  {result['signed'].sum():,}")
print(f"Cases with any petition:  {result['case_protested'].sum():,} parcel rows")

# ── Mean parcels per case by zone ─────────────────────────────────────────────
per_case = result.groupby(["case_number","zone"]).size().unstack(fill_value=0)
print("\nMean adjacent parcels per case:")
print(per_case.mean().round(1))

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "adjacent_parcel_universe.csv")
result.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ── Comparative summary: protested vs never-protested cases ──────────────────
print("\n=== Zone A parcels: protested vs non-protested cases ===")
zone_a = result[result["zone"] == "A_eligible"].copy()
zone_a_summary = zone_a.groupby("case_protested").agg(
    n_parcel_rows=("parcel_id", "count"),
    n_cases_col=("case_number", "nunique"),
    pct_signed=("signed", "mean"),
).round(3)
zone_a_summary["mean_parcels_per_case"] = (zone_a_summary["n_parcel_rows"] / zone_a_summary["n_cases_col"]).round(1)
print(zone_a_summary)
