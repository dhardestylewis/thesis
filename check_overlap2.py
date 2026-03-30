import os
import csv

data_dir = "c:/Users/dhl/data/thesis/thesis/Data/Zoning_Cases/Processed_Data"
multi_csv = os.path.join(data_dir, "multi_parcel_closed_2018_2025.csv")

multis = set()
with open(multi_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        multis.add(row['CASE_NUMBER'])

transcripts_dir = os.path.join(data_dir, "Transcripts")
ts = [f.replace("_transcript.txt", "").replace("_", "/") for f in os.listdir(transcripts_dir) if f.endswith("_transcript.txt")]

overlap = sum(1 for c in ts if c in multis or c.replace("/", "_") in multis)

with open("c:/Users/dhl/data/thesis/thesis/overlap_results.txt", "w") as f:
    f.write(f"Total transcripts: {len(ts)}\nMulti-parcel matches: {overlap}\nNon-multi-parcel (single/other): {len(ts) - overlap}\n")
