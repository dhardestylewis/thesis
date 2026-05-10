import pandas as pd
import urllib.request
import re
import concurrent.futures
import time

index_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\austin_council_meetings_index.csv"
output_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\council_minutes_index.csv"

df = pd.read_csv(index_csv)

# Filter for regular city council meetings, 2009 onwards
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df_cc = df[(df['Meeting_Text'].str.contains('City Council Regular', case=False, na=False)) & (df['Year'] >= 2009)].copy()

print(f"Total Regular City Council Meetings since 2009: {len(df_cc)}", flush=True)

headers = {"User-Agent": "Mozilla/5.0"}
minutes_data = []
errors = 0

def fetch_minutes_link(row):
    url = row['URL']
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
                # Search for the Minutes link
                match = re.search(r'href="(https://services\.austintexas\.gov/edims/document\.cfm\?id=(\d+))"[^>]*>Minutes</a>', html, re.IGNORECASE)
                if match:
                    return {
                        'Year': row['Year'],
                        'Meeting_Text': row['Meeting_Text'],
                        'Meeting_URL': url,
                        'Minutes_URL': match.group(1),
                        'Doc_ID': match.group(2)
                    }
                else:
                    return None
        except Exception:
            time.sleep(1)
    return 'error'

start_time = time.time()
print("Scraping meeting pages for Minutes links...", flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(fetch_minutes_link, row): row for idx, row in df_cc.iterrows()}
    
    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
        result = future.result()
        if result == 'error':
            errors += 1
        elif result is not None:
            minutes_data.append(result)
            
        if i % 50 == 0 or i == len(df_cc):
            print(f"Processed {i}/{len(df_cc)} meetings... Found {len(minutes_data)} Minutes PDFs.", flush=True)

df_out = pd.DataFrame(minutes_data)
df_out.to_csv(output_csv, index=False)

print(f"\nFinished in {time.time() - start_time:.2f} seconds.", flush=True)
print(f"Saved {len(df_out)} minutes links to {output_csv}", flush=True)
