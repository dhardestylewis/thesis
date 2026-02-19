"""Diagnose distribution shift between Backtest (2024) and Forecast (2025)."""
import csv, sys, os
import numpy as np
from collections import Counter

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PANEL_PATH = "Data/Panel/Output/Property_Year_Panel.csv"
EARS_2025 = "Data/Panel/Intermediate/ears_2025_clean.csv"

def get_distribution(values, name):
    print(f"\n--- {name} Distribution ---")
    if not values:
        print("  (empty)")
        return
    if isinstance(values[0], (int, float)):
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std: {np.std(values):.2f}")
        print(f"  Zeros: {values.count(0)} ({100*values.count(0)/len(values):.1f}%)")
    else:
        c = Counter(values)
        for k, v in c.most_common(10):
            print(f"  {k}: {v} ({100*v/len(values):.1f}%)")

# Load 2024 (Backtest)
print("Loading 2024 data (Backtest)...")
vals_2024 = {"market": [], "cat": [], "council": [], "lui": []}
with open(PANEL_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["year"] == "2024" and row.get("ears_matched") == "1":
            try:
                vals_2024["market"].append(float(row.get("market_value", 0)))
            except: pass
            vals_2024["cat"].append(row.get("property_category_code", ""))
            vals_2024["council"].append(row.get("council_district", ""))
            vals_2024["lui"].append(row.get("lui_general_land_use", ""))

# Load 2025 (Forecast) - Simulate the extraction/fill logic
print("Loading 2025 data (Forecast source)...")
vals_2025 = {"market": [], "cat": [], "council": [], "lui": []}
# Pre-load 2024 lookup for fill
lookup_2024 = {}
# (Re-read panel for lookup)
with open(PANEL_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["year"] == "2024":
            lookup_2024[row["standardized_tcad_id"]] = {
                "lui": row.get("lui_general_land_use", ""),
                "council": row.get("council_district", "")
            }

with open(EARS_2025, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row.get("_parcel_id_10", "").strip()
        if not pid: continue
        
        # Market value (check mapped col)
        mkt = row.get("total_market_value", "")
        try: vals_2025["market"].append(float(mkt))
        except: vals_2025["market"].append(0.0)
        
        # Cat
        vals_2025["cat"].append(row.get("property_category_code", ""))
        
        # Fill logic
        lui = ""
        council = ""
        if pid in lookup_2024:
            lui = lookup_2024[pid]["lui"]
            council = lookup_2024[pid]["council"]
        vals_2025["lui"].append(lui)
        vals_2025["council"].append(council)

# Compare
feat_names = ["market", "cat", "council", "lui"]
for f in feat_names:
    print(f"\n{f.upper()} COMPARISON:")
    print(">> 2024 (Backtest Universe):", len(vals_2024[f]))
    get_distribution(vals_2024[f], "2024")
    print(">> 2025 (Forecast Universe):", len(vals_2025[f]))
    get_distribution(vals_2025[f], "2025")
