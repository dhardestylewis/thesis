import os
import pandas as pd

data_dir = r"c:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data"
multi_csv = os.path.join(data_dir, "multi_parcel_closed_2018_2025.csv")

# Load multi-parcel cases
multis = set(pd.read_csv(multi_csv)['CASE_NUMBER'].astype(str))

# Load all transcript files
transcripts_dir = os.path.join(data_dir, "Transcripts")
ts = [f.replace('_transcript.txt', '').replace('_', '/') 
      for f in os.listdir(transcripts_dir) if f.endswith('_transcript.txt')]

overlap = 0
for c in ts:
    if c in multis or c.replace('/', '_') in multis:
        overlap += 1

print(f"Total transcripts: {len(ts)}")
print(f"Multi-parcel matches: {overlap}")
print(f"Non-multi-parcel (single/other): {len(ts) - overlap}")
