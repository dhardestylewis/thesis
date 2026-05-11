import pandas as pd
import numpy as np
import re
from dateutil import parser

# File Paths
PANEL_PATH = r"C:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv"
ADV_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\advanced_geometric_petition_intensity.csv"
TRANSCRIPTS_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\interim\council_transcripts.csv"

print("Loading datasets...")
panel = pd.read_csv(PANEL_PATH, low_memory=False)
adv = pd.read_csv(ADV_PATH, low_memory=False)
transcripts = pd.read_csv(TRANSCRIPTS_PATH, low_memory=False)

panel["period_start"] = pd.to_datetime(panel["period_start"], format="mixed")
panel["period_end"] = panel["period_start"] + pd.Timedelta(days=14)

# Precompute dates for all transcripts
def extract_date(text):
    if pd.isnull(text):
        return None
    # Fix missing spaces after comma: "OCTOBER 15,2009" -> "OCTOBER 15, 2009"
    header = text[:300].replace(",", ", ")
    match = re.search(r'(?:MINUTES|MINUTES(?:\s+FOR)?)(.*?20\d{2})', header, re.IGNORECASE | re.DOTALL)
    if match:
        date_str = match.group(1).replace('\n', ' ').strip()
        try:
            return parser.parse(date_str, fuzzy=True)
        except:
            return None
    return None

print("Extracting dates from 376 Council Minutes PDFs...")
transcripts["Meeting_Date"] = transcripts["Vote_Transcript"].apply(extract_date)

# Identify the 209 cases that were legally protested
protested_adv = adv[adv["unofficial_protest_intensity"] > 0].copy()
protested_cases = protested_adv["case_number"].unique()

# Identify cases that already have petition_pct_this_period > 0 in the panel
existing_protested = panel[panel["petition_pct_this_period"] > 0]["case_number"].unique()
missing_cases = list(set(protested_cases) - set(existing_protested))

print(f"Total Protested Cases: {len(protested_cases)}")
print(f"Cases already mapped: {len(existing_protested)}")
print(f"Cases missing targets: {len(missing_cases)}")

patched_count = 0
fallback_count = 0
not_found_count = 0

for case in missing_cases:
    case_rows = panel[panel["case_number"] == case]
    if len(case_rows) == 0:
        continue
        
    # Regex search for the case number in all transcripts
    # Sometimes cases have .SH or suffixes, so we search the base case
    base_case = str(case).split('.')[0]
    hits = transcripts[transcripts["Vote_Transcript"].str.contains(base_case, na=False, case=False)]
    
    if len(hits) > 0:
        # Get all valid dates and find the earliest (first reading)
        valid_dates = hits["Meeting_Date"].dropna()
        if len(valid_dates) > 0:
            earliest_date = valid_dates.min()
            
            # Map to the biweekly period
            mask = (panel["case_number"] == case) & (panel["period_start"] <= earliest_date) & (panel["period_end"] >= earliest_date)
            if mask.sum() > 0:
                panel.loc[mask, "petition_event"] = 1
                patched_count += 1
                pct_val = protested_adv[protested_adv["case_number"] == case]["unofficial_protest_intensity"].iloc[0]
                panel.loc[mask, "petition_pct_this_period"] = pct_val
                continue
            
    # Fallback: Last tracked period (if not found in transcripts or date out of bounds)
    last_idx = case_rows["period_seq"].idxmax()
    panel.loc[last_idx, "petition_event"] = 1
    pct_val = protested_adv[protested_adv["case_number"] == case]["unofficial_protest_intensity"].iloc[0]
    panel.loc[last_idx, "petition_pct_this_period"] = pct_val
    fallback_count += 1

print(f"\nSuccessfully mapped {patched_count} cases strictly via Council Minutes NLP.")
print(f"Successfully mapped {fallback_count} cases using Right-Censored Fallback (Last Period).")

print("\nRecalculating lagged cumulative petition trackers...")
panel = panel.sort_values(["case_number", "period_seq"])
panel["petition_event"] = panel["petition_event"].fillna(0).astype(int)
panel["petition_pct_this_period"] = panel["petition_pct_this_period"].fillna(0)

# Lagged by 1 period to prevent contemporaneous leakage!
panel["cumulative_petition_events"] = panel.groupby("case_number")["petition_event"].transform(lambda x: x.cumsum().shift(1).fillna(0))
panel["cumulative_petition_pct"] = panel.groupby("case_number")["petition_pct_this_period"].transform(lambda x: x.cumsum().shift(1).fillna(0))

print("\nValidating Year Distribution of petition_event:")
print(panel.groupby("year")["petition_event"].sum())

print("\nSaving patched panel...")
panel.to_csv(PANEL_PATH, index=False)
print("Complete.")
