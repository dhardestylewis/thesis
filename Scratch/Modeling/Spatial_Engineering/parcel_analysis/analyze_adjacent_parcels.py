"""
Analyze Adjacent Parcels
1. Compare signers vs non-signers within protested cases (Zone A).
2. Compare neighbors of protested cases vs neighbors of non-protested cases.
3. Compare Zone A (0-200ft) vs Zone B (200-500ft) for protested cases.
"""
import pandas as pd
import numpy as np

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
ADJ_PATH = rf"{OUT_DIR}\adjacent_parcel_universe.csv"

print("Loading adjacent parcel universe...")
adj = pd.read_csv(ADJ_PATH, low_memory=False)

# Convert land use to numeric if needed, or map to string
# LUI 100=Single Family, 200=Multifamily, 300=Commercial, 400=Office, etc.
def map_lui(code):
    if pd.isna(code): return "Unknown"
    c = int(code)
    if c == 100: return "Single Family"
    if c == 200: return "Multifamily"
    if c == 300: return "Commercial"
    if c == 400: return "Office"
    if c == 500: return "Industrial"
    if c == 600: return "Civic/Public"
    if c == 700: return "Open Space"
    if c == 800: return "Transportation"
    if c == 900: return "Undeveloped"
    return str(c)

adj["land_use_desc"] = adj["lui_general_land_use"].apply(map_lui)
adj["is_single_family"] = (adj["lui_general_land_use"] == 100).astype(int)
adj["is_commercial"] = (adj["lui_general_land_use"] == 300).astype(int)

print("\n" + "="*50)
print("1. PROTESTED CASES: SIGNERS VS NON-SIGNERS (Zone A)")
print("="*50)
protested_zone_a = adj[(adj["case_protested"] == 1) & (adj["zone"] == "A_eligible")]
print(f"Total eligible parcels in protested cases: {len(protested_zone_a):,}")

signer_comp = protested_zone_a.groupby("signed").agg(
    count=("parcel_id", "count"),
    pct_single_family=("is_single_family", "mean"),
    pct_commercial=("is_commercial", "mean"),
    median_lot_size_sqft=("lui_shape_area", "median"),
).round(3)
signer_comp.index = ["Did Not Sign", "Signed"]
print(signer_comp.to_string())

print("\n" + "="*50)
print("2. ALL ZONING CASES: NEIGHBORS OF PROTESTED VS NON-PROTESTED (Zone A)")
print("="*50)
zone_a = adj[adj["zone"] == "A_eligible"]
case_comp = zone_a.groupby("case_protested").agg(
    count=("parcel_id", "count"),
    pct_single_family=("is_single_family", "mean"),
    pct_commercial=("is_commercial", "mean"),
    median_lot_size_sqft=("lui_shape_area", "median"),
).round(3)
case_comp.index = ["Non-Protested Cases", "Protested Cases"]
print(case_comp.to_string())

print("\n" + "="*50)
print("3. PROTESTED CASES: ZONE A (Eligible) VS ZONE B (Ineligible 200-500ft)")
print("="*50)
protested_all_zones = adj[adj["case_protested"] == 1]
zone_comp = protested_all_zones.groupby("zone").agg(
    count=("parcel_id", "count"),
    pct_single_family=("is_single_family", "mean"),
    pct_commercial=("is_commercial", "mean"),
    median_lot_size_sqft=("lui_shape_area", "median"),
).round(3)
print(zone_comp.to_string())

print("\n" + "="*50)
print("4. SPATIAL DIFFUSION OF PROTEST")
print("="*50)
print("If the legal threshold was 500ft instead of 200ft, how many single-family")
print("homes are currently disenfranchised (in Zone B) near protested cases?")
sf_zone_b = protested_all_zones[(protested_all_zones["zone"] == "B_ineligible") & (protested_all_zones["is_single_family"] == 1)]
print(f"Single-family homes in Zone B (200-500ft) of protested cases: {len(sf_zone_b):,}")
avg_disenfranchised = len(sf_zone_b) / protested_all_zones["case_number"].nunique()
print(f"Average single-family homes in 200-500ft ring per protested case: {avg_disenfranchised:.1f}")

# Also output a clean artifact
with open(rf"{OUT_DIR}\adjacent_parcel_analysis.md", "w") as f:
    f.write("# Adjacent Parcel Analysis\n\n")
    f.write("## 1. Who Signs Petitions?\n")
    f.write("Comparing adjacent parcels (0-200ft) that signed vs didn't sign within protested cases:\n\n")
    f.write(signer_comp.to_markdown())
    f.write("\n\n## 2. Protested vs Non-Protested Neighborhoods\n")
    f.write(case_comp.to_markdown())
    f.write("\n\n## 3. The 200ft vs 500ft Ring\n")
    f.write(zone_comp.to_markdown())
    f.write(f"\n\n*Average disenfranchised single-family homes (200-500ft) per protested case: {avg_disenfranchised:.1f}*\n")
