"""
Phase 1: Asynchronous PDF Acquisition
Downloads 10,000+ City of Austin Council and Commission meeting agendas and transcripts.
Handles rate-limiting and asynchronous networking to build the local Data/Commission_PDFs corpus.
"""

import pandas as pd
import urllib.request
import concurrent.futures
import time
import os
import re

plan_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\planning_commission_index.csv"
zap_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\zoning_platting_commission_index.csv"
output_dir = r"c:\Users\dhl\data\Thesis\thesis\Data\Commission_PDFs"
os.makedirs(output_dir, exist_ok=True)

df_plan = pd.read_csv(plan_csv)
df_zap = pd.read_csv(zap_csv)
df_all = pd.concat([df_plan, df_zap], ignore_index=True)

df_target = df_all.drop_duplicates(subset=['Doc_URL']).copy()
print(f"Total targeted documents to download: {len(df_target)}", flush=True)

download_tasks = []
for idx, row in df_target.iterrows():
    clean_name = re.sub(r'[^A-Za-z0-9_\-\. ]', '_', str(row['Doc_Text']))
    clean_name = clean_name[:150]
    filename = f"{row['Year']}_{row['Doc_ID']}_{clean_name}.pdf"
    file_path = os.path.join(output_dir, filename)
    download_tasks.append((row['Doc_URL'], file_path))

downloaded = 0
skipped = 0
errors = 0

def download_file(task):
    url, path = task
    if os.path.exists(path):
        return 'skipped'
    
    for _ in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(path, "wb") as f:
                    f.write(response.read())
            return 'downloaded'
        except Exception:
            time.sleep(1)
            continue
            
    return 'error'

start_time = time.time()
print("Starting concurrent download with 20 workers...", flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(download_file, task): task for task in download_tasks}
    
    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
        result = future.result()
        if result == 'skipped':
            skipped += 1
        elif result == 'downloaded':
            downloaded += 1
        else:
            errors += 1
            
        if i % 100 == 0 or i == len(download_tasks):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"[{i}/{len(download_tasks)}] Downloaded: {downloaded} | Skipped: {skipped} | Errors: {errors} | Rate: {rate:.1f} files/sec", flush=True)

print(f"\nFinished in {time.time() - start_time:.2f} seconds.", flush=True)
print(f"Total Downloaded: {downloaded}", flush=True)
print(f"Total Skipped: {skipped}", flush=True)
print(f"Total Errors: {errors}", flush=True)
