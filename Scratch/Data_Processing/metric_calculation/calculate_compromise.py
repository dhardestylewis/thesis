import pandas as pd
import numpy as np

# Load the data
print("Loading datasets...")
mr = pd.read_csv(r"C:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv", usecols=["case_number", "Requested_Zoning", "Final_Zoning"])
panel = pd.read_csv(r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv", low_memory=False)

# Get the final state of each case from the panel
cases = panel[panel["period_seq"] == 1][["case_number", "year", "proposed_max_height_ft", "existing_max_height_ft", "label_valid_protest"]].copy()

# Merge the final zoning strings
df = cases.merge(mr, on="case_number", how="left")

# Clean the final zoning
df["Final_Zoning"] = df["Final_Zoning"].fillna("UNKNOWN").astype(str)

def get_height_limit(zone_string):
    if pd.isna(zone_string) or zone_string in ["UNKNOWN", "NO", "WITHDRAWN"]:
        return np.nan
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

# Map the final approved height
df["approved_max_height_ft"] = df["Final_Zoning"].apply(get_height_limit)

# Drop missing
df = df.dropna(subset=["proposed_max_height_ft", "approved_max_height_ft"])

# Calculate the compromise margin (+ means they successfully forced a reduction from the proposal)
df["compromise_margin_ft"] = df["proposed_max_height_ft"] - df["approved_max_height_ft"]

print("\n--- Compromise Analysis ---")
print(f"Total cases with calculable compromise margins: {len(df)}")

# Filter to cases where a compromise actually happened (approved height is less than proposed height)
compromised = df[df["compromise_margin_ft"] > 0]
print(f"Total cases that were successfully 'downzoned' from the proposal: {len(compromised)} ({len(compromised)/len(df)*100:.1f}%)")

if len(compromised) > 0:
    print(f"\nAverage height shaved off the proposal: {compromised['compromise_margin_ft'].mean():.1f} ft")
    
    print("\nImpact of Formal Protests on Compromise:")
    protested = compromised[compromised["label_valid_protest"] == 1]
    unprotested = compromised[compromised["label_valid_protest"] == 0]
    
    print(f"  - Avg reduction WHEN formally protested: {protested['compromise_margin_ft'].mean():.1f} ft (N={len(protested)})")
    print(f"  - Avg reduction when NOT protested:      {unprotested['compromise_margin_ft'].mean():.1f} ft (N={len(unprotested)})")

print("\nTop 5 Largest Height Reductions from Proposal:")
print(compromised.sort_values("compromise_margin_ft", ascending=False)[["case_number", "year", "Requested_Zoning", "Final_Zoning", "proposed_max_height_ft", "approved_max_height_ft", "compromise_margin_ft"]].head(5).to_string(index=False))

print("\n--- Temporal Breakdown of Reductions ---")
# Group by era (e.g., Pre-2018 vs 2018-Present)
compromised["era"] = np.where(compromised["year"] >= 2018, "Recent (2018-2024)", "Historical (2007-2017)")
temporal = compromised.groupby("era").agg(
    Cases=("case_number", "count"),
    Avg_Reduction_ft=("compromise_margin_ft", "mean"),
    Max_Reduction_ft=("compromise_margin_ft", "max")
).reset_index()
print(temporal.to_string(index=False))
