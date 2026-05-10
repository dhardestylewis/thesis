import pandas as pd
import re

# 1. Load the thesis zoning cases
df_cases = pd.read_csv('c:/Users/dhl/data/Thesis/thesis/Data/Zoning_Cases/Processed_Data/CSV/zoning_land_use_merged_data.csv')
target_cases = set(df_cases['case_number'].dropna().astype(str).str.upper().unique())
print(f"Total unique zoning cases in thesis dataset: {len(target_cases)}")

# 2. Load the newly scraped commission indexes
df_plan = pd.read_csv('c:/Users/dhl/data/Thesis/thesis/Data/planning_commission_index.csv')
df_zap = pd.read_csv('c:/Users/dhl/data/Thesis/thesis/Data/zoning_platting_commission_index.csv')
df_scraped = pd.concat([df_plan, df_zap], ignore_index=True)

# 3. Naive substring matching (fast enough for ~15k documents and a few thousand cases)
matched_docs = 0
matched_cases = set()

# Pre-calculate a cleaned text column
df_scraped['Clean_Text'] = df_scraped['Doc_Text'].astype(str).str.upper()

# Filter target cases to only those from 2009 onwards (since our scraped docs are 2009+)
# Austin cases use the year in the middle segment: C14-2015-0123
modern_cases = [c for c in target_cases if re.search(r'-(20[0-2][0-9])-|-([0-2][0-9])-', c)]
# Actually, let's just search all of them

for case in target_cases:
    # Some cases in the dataset might be like 'C14-2015-0012' but written as 'C14-15-0012'
    # We will just search the exact string first
    matches = df_scraped['Clean_Text'].str.contains(case, regex=False)
    if matches.any():
        matched_cases.add(case)
        matched_docs += matches.sum()

print(f"Number of target zoning cases found in scraped documents: {len(matched_cases)} out of {len(target_cases)}")
print(f"Total scraped documents mapped to a target zoning case: {matched_docs} out of {len(df_scraped)}")
