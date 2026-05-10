import os
import re
import pandas as pd

BASE = r"c:\Users\dhl\data\Thesis\thesis\Data"

TRANSCRIPTS_CSV = os.path.join(BASE, "commission_transcripts.csv")
PLAN_INDEX_CSV = os.path.join(BASE, "planning_commission_index.csv")
ZON_INDEX_CSV = os.path.join(BASE, "zoning_platting_commission_index.csv")
OUT_CSV = os.path.join(BASE, "commission_agendas_cases.csv")

print("1. Loading Commission Indices...")
plan_idx = pd.read_csv(PLAN_INDEX_CSV, usecols=["Meeting_Date", "Doc_ID"])
zon_idx = pd.read_csv(ZON_INDEX_CSV, usecols=["Meeting_Date", "Doc_ID"])

# Combine indices
all_idx = pd.concat([plan_idx, zon_idx]).drop_duplicates(subset=["Doc_ID"])
all_idx["Doc_ID"] = all_idx["Doc_ID"].astype(str)

print(f"   Loaded {len(all_idx)} unique Document IDs with Meeting Dates.")

print("\n2. Parsing Transcripts for Case Numbers...")
# We chunk the transcripts because 171MB of text could cause memory issues if exploded completely
chunksize = 1000
case_pattern = re.compile(r'(C(?:14|814|8J?)-\d{2,4}-\d{4}(?:\.\d+[A-Z]?)?)')

rows = []
for chunk in pd.read_csv(TRANSCRIPTS_CSV, chunksize=chunksize):
    # Extract Doc_ID from Filename (e.g. 2009_136932_Agenda_52KB_.pdf -> 136932)
    chunk["Doc_ID"] = chunk["Filename"].str.extract(r'^\d+_(\d+)_')[0]
    
    # Drop rows without text or Doc_ID
    chunk = chunk.dropna(subset=["Raw_Text", "Doc_ID"])
    
    for _, row in chunk.iterrows():
        text = str(row["Raw_Text"])
        cases = set(case_pattern.findall(text))
        for case in cases:
            rows.append({
                "Doc_ID": row["Doc_ID"],
                "Case_Number": case
            })

parsed_df = pd.DataFrame(rows)
print(f"   Found {len(parsed_df)} total case mentions across documents.")

print("\n3. Mapping to Meeting Dates...")
parsed_df = parsed_df.merge(all_idx, on="Doc_ID", how="inner")

# Clean Meeting Dates
parsed_df["Meeting_Date"] = pd.to_datetime(
    parsed_df["Meeting_Date"].str.extract(r'^([A-Za-z]+ \d+,?\s+\d{4})')[0].str.strip(),
    format="mixed", errors="coerce"
)
parsed_df = parsed_df.dropna(subset=["Meeting_Date"])

# Group by Case_Number and Meeting_Date to remove duplicate mentions in the same document
final_df = parsed_df[["Case_Number", "Meeting_Date"]].drop_duplicates()

final_df = final_df.rename(columns={"Case_Number": "case_number", "Meeting_Date": "meeting_date"})
final_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved {OUT_CSV}")
print(f"Total Unique Commission Hearings: {len(final_df)}")
print(f"Total Cases with Commission Hearings: {final_df['case_number'].nunique()}")
