import pandas as pd
import numpy as np

PANEL_PATH = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"

print("1. Loading Bi-Weekly Panel...")
df = pd.read_csv(PANEL_PATH, low_memory=False)

leakage_vars = [
    "cumulative_petition_events",
    "cumulative_petition_count",
    "cumulative_petition_pct",
    "cumulative_council_hearings",
    "cumulative_commission_hearings",
    "Remand_Count"
]

print("2. Calculating Lag-1 Features by Case...")
df = df.sort_values(["case_number", "period_seq"])

for var in leakage_vars:
    # Shift by 1 period within each case group
    df[f"lag1_{var}"] = df.groupby("case_number")[var].shift(1)
    # Fill NAs resulting from the shift with 0 (no events happened before period 1)
    df[f"lag1_{var}"] = df[f"lag1_{var}"].fillna(0)

# Drop the leaked variables from the panel to ensure no one ever uses them again
df = df.drop(columns=leakage_vars)

print("3. Saving Fixed Panel...")
df.to_csv(PANEL_PATH, index=False)
print(f"Successfully remediated target leakage. Panel shape: {df.shape}")
