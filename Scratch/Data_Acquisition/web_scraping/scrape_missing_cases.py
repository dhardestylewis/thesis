import pandas as pd
import urllib.request
import re
import concurrent.futures
import time
import os

index_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\austin_council_meetings_index.csv"
output_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\council_agendas_missing_cases.csv"

df = pd.read_csv(index_csv)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df_cc = df[(df['Meeting_Text'].str.contains('City Council Regular', case=False, na=False)) & (df['Year'] >= 2009)].copy()

headers = {"User-Agent": "Mozilla/5.0"}
case_data = []

def fetch_agenda_cases(row):
    url = row['URL']
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
            # Find missing cases: C814, NPA, C14H, C17
            matches = re.findall(r'((?:C814|NPA|C14H|C17)-\d{2,4}-\d{2,4}(?:\.[a-zA-Z0-9]+)?)', html, re.IGNORECASE)
            unique_cases = list(set([m.upper() for m in matches]))
            
            records = []
            for case in unique_cases:
                records.append({
                    'Year': row['Year'],
                    'Meeting_Date': row['Meeting_Text'],
                    'Meeting_URL': url,
                    'Case_Number': case
                })
            return records
    except Exception as e:
        return []

start_time = time.time()
print(f"Scraping {len(df_cc)} Agendas for Missing Cases (C814, NPA, C14H, C17)...", flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(fetch_agenda_cases, row): row for idx, row in df_cc.iterrows()}
    
    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
        results = future.result()
        case_data.extend(results)

df_out = pd.DataFrame(case_data)
df_out.to_csv(output_csv, index=False)
elapsed = time.time() - start_time

print(f"Finished in {elapsed:.2f} seconds.", flush=True)
print(f"Extracted {len(df_out)} missing case appearances.", flush=True)
