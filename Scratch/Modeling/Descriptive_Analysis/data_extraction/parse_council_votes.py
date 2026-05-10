"""
Council Vote Transcript Parsing
Parses actual X-Y council votes from the transcript texts to determine 
if protested cases face more divided (split) council votes than non-protested cases.
"""
import pandas as pd
import numpy as np
import re

OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"
VOTES_TRANSCRIPT = r"C:\Users\dhl\data\Thesis\thesis\Data\interim\zoning_cases_with_council_votes.csv"
PET_INTENSITY = rf"{OUT_DIR}\petition_intensity.csv"

# Load data
df = pd.read_csv(VOTES_TRANSCRIPT, low_memory=False)
pet = pd.read_csv(PET_INTENSITY)

# Standardize case numbers
df["Case_Number"] = df["Case_Number"].str.strip()
pet["case_number"] = pet["case_number"].str.strip()

# Extract X-Y votes from transcripts using regex
# Looks for patterns like "on a 7-0 vote" or "11-0 vote"
vote_pattern = re.compile(r'\b(\d{1,2})-(\d{1,2})\s*vote\b', re.IGNORECASE)

parsed_votes = []
for _, row in df.iterrows():
    text = str(row["Vote_Transcript"])
    matches = vote_pattern.findall(text)
    if matches:
        for match in matches:
            yes_votes, no_votes = int(match[0]), int(match[1])
            parsed_votes.append({
                "Case_Number": row["Case_Number"],
                "Meeting_Date": row["Meeting_Date"],
                "Yes_Votes": yes_votes,
                "No_Votes": no_votes,
                "Total_Votes": yes_votes + no_votes
            })

parsed_df = pd.DataFrame(parsed_votes)

# Filter out weird errors (Austin council is 11 members, formerly 7 before 10-1)
parsed_df = parsed_df[(parsed_df["Total_Votes"] >= 3) & (parsed_df["Total_Votes"] <= 11)]

# Get the most contentious vote (highest No votes) per case
contention = parsed_df.groupby("Case_Number").agg(
    max_no_votes=("No_Votes", "max"),
    avg_no_votes=("No_Votes", "mean"),
    total_vote_events=("Meeting_Date", "count")
).reset_index()

# Merge with protest data
all_cases = df[["Case_Number"]].drop_duplicates()
all_cases = all_cases.merge(contention, on="Case_Number", how="left")
all_cases["max_no_votes"] = all_cases["max_no_votes"].fillna(0) # assume unanimous if no split mentioned
all_cases = all_cases.merge(pet[["case_number", "petition_n_parcels"]], left_on="Case_Number", right_on="case_number", how="left")
all_cases["any_protest"] = (all_cases["petition_n_parcels"] > 0).astype(int)

# Create vote categories
def split_category(no_votes):
    if pd.isna(no_votes): return "Unknown"
    if no_votes == 0: return "Unanimous (0 Nays)"
    if no_votes in [1, 2]: return "Minor Split (1-2 Nays)"
    if no_votes >= 3: return "Contentious (3+ Nays)"
    return "Unknown"

all_cases["vote_split"] = all_cases["max_no_votes"].apply(split_category)

print("\n" + "="*60)
print("COUNCIL VOTE CONTENTION (Extracted from Meeting Transcripts)")
print("="*60)

# Filter to cases where we actually parsed a vote
cases_with_votes = all_cases[all_cases["total_vote_events"] > 0]

summary = cases_with_votes.groupby("any_protest").agg(
    cases_with_recorded_vote=("Case_Number", "count"),
    avg_nay_votes=("avg_no_votes", "mean"),
    max_nay_votes=("max_no_votes", "max")
).round(2)
summary.index = ["No Protest", "Any Protest"]
print(summary.to_string())

print("\nDistribution of Split Votes:")
xtab = pd.crosstab(cases_with_votes["any_protest"], cases_with_votes["vote_split"], normalize="index") * 100
xtab.index = ["No Protest", "Any Protest"]
print(xtab.round(1).to_string())

with open(rf"{OUT_DIR}\council_vote_transcript_analysis.md", "w") as f:
    f.write("# Council Vote Splits (Parsed from Transcripts)\n\n")
    f.write(summary.to_markdown())
    f.write("\n\n### Vote Margin Distribution (%)\n\n")
    f.write(xtab.round(1).to_markdown())
