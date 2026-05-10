import os
import csv

import sys, os
ROOT_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR_PATH not in sys.path: sys.path.append(ROOT_DIR_PATH)
from pipeline.config.paths import ZONING_CASES_DIR

data_dir = ZONING_CASES_DIR / "Processed_Data"
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
