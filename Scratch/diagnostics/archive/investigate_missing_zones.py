import pandas as pd
import re

model_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv"
df_model = pd.read_csv(model_csv)
df_comm = pd.read_csv(r"c:\Users\dhl\data\Thesis\thesis\Data\commission_transcripts.csv")

# Find cases that failed to extract zoning
missing_cases = set(df_model[df_model['Requested_Zoning'].isna()]['Core_Case'].dropna().unique())

print(f"Total missing cases: {len(missing_cases)}")

count = 0
for t in df_comm['Raw_Text'].dropna():
    text_str = str(t).upper()
    for case in list(missing_cases)[:100]: # check a subset to be fast
        idx = text_str.find(case)
        if idx != -1:
            print(f"\n--- MISSING CASE: {case} ---")
            start = max(0, idx - 100)
            end = min(len(text_str), idx + 400)
            print(text_str[start:end])
            count += 1
            missing_cases.remove(case)
            
    if count >= 30:
        break
