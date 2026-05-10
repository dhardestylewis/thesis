import urllib.request
import time
import random
import statistics
import os

doc_ids = [
    "472570", "472571", "472572", "472573", "472575", 
    "472574", "472576", "472577", "472578", "472579",
    "472580", "472581", "472582", "472583", "472584"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

output_dir = "downloaded_samples"
os.makedirs(output_dir, exist_ok=True)

download_times = []
download_speeds = [] 

print(f"Starting download test for {len(doc_ids)} PDFs with humanized delays...")

for i, doc_id in enumerate(doc_ids):
    url = f"https://services.austintexas.gov/edims/document.cfm?id={doc_id}"
    
    if i > 0:
        delay = random.uniform(2.0, 6.0)
        print(f"Sleeping for {delay:.2f} seconds...")
        time.sleep(delay)
        
    print(f"Downloading {url} ...")
    start_time = time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            end_time = time.time()
            
            content_size = len(content)
            duration = end_time - start_time
            speed = content_size / duration if duration > 0 else 0
            
            download_times.append(duration)
            download_speeds.append(speed)
            
            file_path = os.path.join(output_dir, f"{doc_id}.pdf")
            with open(file_path, "wb") as f:
                f.write(content)
                
            print(f"Success: {content_size/1024:.2f} KB in {duration:.2f} seconds ({speed/1024/1024:.2f} MB/s)")
            
    except Exception as e:
        print(f"Failed to download {url}: {e}")

if download_speeds:
    mean_speed = statistics.mean(download_speeds) / 1024 / 1024
    median_speed = statistics.median(download_speeds) / 1024 / 1024
    mean_time = statistics.mean(download_times)
    
    print("\n--- Download Statistics ---")
    print(f"Total Successful Downloads: {len(download_speeds)}")
    print(f"Mean Download Speed: {mean_speed:.2f} MB/s")
    print(f"Median Download Speed: {median_speed:.2f} MB/s")
    print(f"Mean Time per File: {mean_time:.2f} seconds")
    print(f"Estimated time to download 1,000 files (excluding sleeps): {mean_time * 1000 / 60:.2f} minutes")
    print(f"Estimated time to download 1,000 files (WITH 4s humanized sleeps): {(mean_time + 4) * 1000 / 60 / 60:.2f} hours")
else:
    print("No downloads were successful.")
