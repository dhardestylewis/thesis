"""
Detailed Status and Council Vote Analysis
Analyzes raw DETAILED_STATUS from case_master.csv and council vote splits from council_vote_summary.csv
for protested vs non-protested cases.
"""
import pandas as pd
import numpy as np

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
CASE_MASTER = r"C:\Users\dhl\data\Thesis\thesis\Data\Warehouse_As_Of\Build\case_master.csv"
VOTES = r"C:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\council_vote_summary.csv"
PET_INTENSITY = rf"{OUT_DIR}\petition_intensity.csv"

# Load datasets
cm = pd.read_csv(CASE_MASTER, low_memory=False)
cm["CASE_NUMBER"] = cm["CASE_NUMBER"].str.strip()
votes = pd.read_csv(VOTES, low_memory=False)
votes["CASE_NUMBER"] = votes["CASE_NUMBER"].str.strip()

pet = pd.read_csv(PET_INTENSITY)
pet = pet.rename(columns={"case_number": "CASE_NUMBER"})
pet["any_protest"] = (pet["petition_n_parcels"] > 0).astype(int)

# 1. Raw Detailed Status Breakdown
df_status = cm[["CASE_NUMBER", "DETAILED_STATUS"]].drop_duplicates("CASE_NUMBER").merge(
    pet[["CASE_NUMBER", "any_protest"]], on="CASE_NUMBER", how="left"
)
df_status["any_protest"] = df_status["any_protest"].fillna(0)

# Map raw detailed statuses to clean categories
def clean_status(s):
    if pd.isna(s): return "Unknown"
    s = s.lower()
    if "approved" in s: return "Approved"
    if "withdrawn" in s: return "Withdrawn"
    if "closed" in s or "void" in s or "expired" in s: return "Closed/Dead"
    if "active" in s or "pending" in s or "review" in s or "open" in s: return "Active/Pending"
    if "denied" in s: return "Denied"
    return "Other"

df_status["Status_Category"] = df_status["DETAILED_STATUS"].apply(clean_status)

print("\n" + "="*60)
print("1. ACTUAL CASE STATUS (From Raw DETAILED_STATUS)")
print("="*60)
status_xtab = pd.crosstab(df_status["any_protest"], df_status["Status_Category"], normalize="index") * 100
status_xtab.index = ["No Protest", "Any Protest"]
print("Percentage of cases falling into each actual status bucket:")
print(status_xtab.round(1).to_string())

print("\nRaw counts:")
counts_xtab = pd.crosstab(df_status["any_protest"], df_status["Status_Category"])
counts_xtab.index = ["No Protest", "Any Protest"]
print(counts_xtab.to_string())

# 2. Council Vote Splits
df_votes = votes.merge(pet[["CASE_NUMBER", "any_protest"]], on="CASE_NUMBER", how="left")
df_votes["any_protest"] = df_votes["any_protest"].fillna(0)
# Unanimous is true/false. Let's make it int
df_votes["unanimous"] = df_votes["unanimous"].astype(float)
df_votes["nay_votes"] = pd.to_numeric(df_votes["final_vote_no"], errors="coerce").fillna(0)

print("\n" + "="*60)
print("2. COUNCIL VOTE SPLITS (Among Cases that Reached Council Vote)")
print("="*60)

vote_comp = df_votes.groupby("any_protest").agg(
    n_council_cases=("CASE_NUMBER", "count"),
    pct_unanimous_vote=("unanimous", "mean"),
    avg_nay_votes=("nay_votes", "mean"),
    max_nay_votes=("nay_votes", "max")
).round(3)
vote_comp.index = ["No Protest", "Any Protest"]
vote_comp["pct_unanimous_vote"] = vote_comp["pct_unanimous_vote"] * 100
print(vote_comp.to_string())

# Look at specific split types
def vote_margin(row):
    yes = pd.to_numeric(row["final_vote_yes"], errors="coerce")
    no = pd.to_numeric(row["final_vote_no"], errors="coerce")
    if pd.isna(yes) or pd.isna(no): return "Unknown"
    if no == 0: return "Unanimous"
    if no in [1, 2]: return "Minor Opposition (1-2 Nays)"
    if no >= 3: return "Contentious (3+ Nays)"
    return "Unknown"

df_votes["vote_margin"] = df_votes.apply(vote_margin, axis=1)

margin_xtab = pd.crosstab(df_votes["any_protest"], df_votes["vote_margin"], normalize="index") * 100
margin_xtab.index = ["No Protest", "Any Protest"]
print("\nDistribution of Vote Contention (%):")
print(margin_xtab.round(1).to_string())

# Output as artifact
with open(rf"{OUT_DIR}\council_vote_analysis.md", "w") as f:
    f.write("# Actual Outcomes & Council Vote Analysis\n\n")
    f.write("## 1. Actual Resolution Status (From Open Data DETAILED_STATUS)\n\n")
    f.write(status_xtab.round(1).to_markdown())
    f.write("\n\n## 2. Council Vote Splits (Contentiousness)\n\n")
    f.write(vote_comp.to_markdown())
    f.write("\n\n### Vote Margin Distribution (%)\n\n")
    f.write(margin_xtab.round(1).to_markdown())
