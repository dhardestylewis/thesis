import pandas as pd
import numpy as np

PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"

print("1. Loading Bi-Weekly Panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)

macro_vars = [
    "mortgage_rate_30yr",
    "treasury_10yr_yield",
    "fed_funds_rate",
    "local_unemployment_rate"
]

print("2. Extracting Period 1 (Filing) Baselines...")
baselines = df[df["period_seq"] == 1][["case_number"] + macro_vars].copy()

# Rename to _filing
rename_dict = {m: f"{m}_filing" for m in macro_vars}
baselines = baselines.rename(columns=rename_dict)

print("3. Merging Baselines and Calculating Shock Deltas...")
df = df.merge(baselines, on="case_number", how="left")

for m in macro_vars:
    df[f"{m}_filing_delta"] = df[m] - df[f"{m}_filing"]

# Drop the raw _filing columns to keep it clean, we only need the deltas
df = df.drop(columns=list(rename_dict.values()))

print("4. Saving Updated Panel...")
df.to_csv(PANEL_PATH, index=False)
print(f"Successfully engineered filing deltas. Panel shape: {df.shape}")
