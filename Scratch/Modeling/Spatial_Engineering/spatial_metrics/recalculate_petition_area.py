"""
Recalculate True Petition Intensity
The raw 'area_pct' from the city data is flawed because the denominator (total buffer area)
was computed inconsistently.
This script rigorously calculates the true legal petition percentage:
Total Area of Signers (in 200ft buffer) / Total Area of ALL Parcels (in 200ft buffer)
"""
import pandas as pd
import numpy as np

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
ADJ_PATH = rf"{OUT_DIR}\adjacent_parcel_universe.csv"
REC_PATH = rf"{OUT_DIR}\recovered_petitions.csv"
PET_OUT = rf"{OUT_DIR}\petition_intensity_corrected.csv"

# 1. Load Adjacent Universe
adj = pd.read_csv(ADJ_PATH, low_memory=False)

# 2. Filter to just the legally eligible 200ft buffer (Zone A)
zone_a = adj[adj["zone"] == "A_eligible"].copy()

# 3. Denominator: Total parcel area inside the 200ft buffer per case
denominator = zone_a.groupby("case_number").agg(
    total_eligible_sqft=("lui_shape_area", "sum")
).reset_index()

# 4. Numerator: Total area of parcels that signed
# The adjacent universe already has a 'signed' flag from my earlier spatial join!
# Wait, let me make sure it does.
# If not, I'll merge it with recovered_petitions.csv.
rec = pd.read_csv(REC_PATH)
rec["tcad_str"] = rec["tcad_id"].astype(str).str.strip()
zone_a["tcad_str"] = zone_a["parcel_id"].astype(str).str.strip()

# Create a set of signers per case to handle duplicates safely
signers_by_case = rec.groupby("case_number")["tcad_str"].apply(set).to_dict()

def is_signer(row):
    case = row["case_number"]
    tcad = row["tcad_str"]
    if case in signers_by_case and tcad in signers_by_case[case]:
        return 1
    return 0

zone_a["signed_verified"] = zone_a.apply(is_signer, axis=1)

numerator = zone_a[zone_a["signed_verified"] == 1].groupby("case_number").agg(
    signer_sqft=("lui_shape_area", "sum"),
    petition_n_parcels=("parcel_id", "nunique")
).reset_index()

# 5. Compute the True Percentage
true_pet = denominator.merge(numerator, on="case_number", how="left")
true_pet["signer_sqft"] = true_pet["signer_sqft"].fillna(0)
true_pet["petition_n_parcels"] = true_pet["petition_n_parcels"].fillna(0)

true_pet["true_petition_pct"] = (true_pet["signer_sqft"] / true_pet["total_eligible_sqft"]) * 100
true_pet["true_petition_pct"] = true_pet["true_petition_pct"].fillna(0)

# 6. Flag the valid protests (>= 20%)
true_pet["label_valid_protest"] = (true_pet["true_petition_pct"] >= 20.0).astype(int)
true_pet["running_var"] = true_pet["true_petition_pct"] - 20.0

# 7. Compare the Old flawed pct to the New rigorous pct
old_pet = pd.read_csv(rf"{OUT_DIR}\petition_intensity.csv")
comp = true_pet.merge(old_pet[["case_number", "petition_area_pct_raw"]], on="case_number", how="inner")

print("\n=== TRUE PETITION PERCENTAGE CALCULATION ===")
print("Comparing Flawed Raw Data ('petition_area_pct_raw') to Rigorous Spatial Calculation ('true_petition_pct')")
print(comp[["case_number", "petition_area_pct_raw", "true_petition_pct"]].sort_values("petition_area_pct_raw", ascending=False).head(15).round(2).to_string())

# Save the corrected intensity
true_pet.to_csv(PET_OUT, index=False)
print(f"\nCorrected petition intensity saved to {PET_OUT}")

print(f"Number of legally valid protests (>=20%) using TRUE area: {true_pet['label_valid_protest'].sum()}")
