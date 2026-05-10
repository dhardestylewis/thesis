import pandas as pd
import urllib.request
import concurrent.futures
import time
import os

index_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\council_minutes_index.csv"
output_dir = r"c:\Users\dhl\data\Thesis\thesis\Data\Council_Minutes_PDFs"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(index_csv)
df = df.dropna(subset=['Minutes_URL'])
df = df.drop_duplicates(subset=['Minutes_URL'])

download_tasks = []
for idx, row in df.iterrows():
    filename = f"{row['Year']}_{row['Doc_ID']}_Minutes.pdf"
    file_path = os.path.join(output_dir, filename)
    download_tasks.append((row['Minutes_URL'], file_path))

downloaded = 0
skipped = 0
errors = 0

def download_file(task):
    url, path = task
    if os.path.exists(path):
        return 'skipped'
    
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            # 45s timeout because minutes PDFs can be very large
            with urllib.request.urlopen(req, timeout=45) as response:
                with open(path, "wb") as f:
                    f.write(response.read())
            return 'downloaded'
        except Exception:
            time.sleep(1)
            continue
    return 'error'

start_time = time.time()
print(f"Starting concurrent download of {len(download_tasks)} Minutes PDFs...", flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_file, task): task for task in download_tasks}
    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
        res = future.result()
        if res == 'skipped': skipped += 1
        elif res == 'downloaded': downloaded += 1
        else: errors += 1
        
        if i % 25 == 0 or i == len(download_tasks):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"[{i}/{len(download_tasks)}] Downloaded: {downloaded} | Skipped: {skipped} | Errors: {errors} | Rate: {rate:.2f} files/sec", flush=True)

print(f"\nFinished in {time.time() - start_time:.2f} seconds.", flush=True)
