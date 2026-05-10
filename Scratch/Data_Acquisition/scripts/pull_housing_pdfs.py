import urllib.request
import re
import os
import time
import random

urls = [
    'https://www.austintexas.gov/council/2026/20260127-hpc',
    'https://www.austintexas.gov/council/2026/20260210-hpc'
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

output_dir = r"c:\Users\dhl\data\Thesis\thesis\Data\Housing_PDFs"
os.makedirs(output_dir, exist_ok=True)

all_docs = []

for url in urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            docs = re.findall(r'document\.cfm\?id=(\d+)', html)
            all_docs.extend(docs)
    except Exception as e:
        print(f"Error fetching {url}: {e}")

all_docs = list(set(all_docs))
# Take 5 samples
sample_docs = all_docs[:5]

print(f"Found {len(all_docs)} unique documents across 2 recent Housing meetings. Downloading {len(sample_docs)} samples...")

downloaded_paths = []

for i, doc_id in enumerate(sample_docs):
    doc_url = f"https://services.austintexas.gov/edims/document.cfm?id={doc_id}"
    print(f"Downloading {doc_url}...")
    
    if i > 0:
        time.sleep(random.uniform(2.0, 5.0))
        
    try:
        req = urllib.request.Request(doc_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            # Look for content-disposition to see if there's a real filename
            cd = response.headers.get('Content-Disposition')
            filename = f"HPC_doc_{doc_id}.pdf"
            if cd:
                fname_match = re.search(r'filename="([^"]+)"', cd)
                if fname_match:
                    filename = fname_match.group(1).replace(" ", "_").replace("/", "-")
            
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "wb") as f:
                f.write(content)
            downloaded_paths.append(file_path)
            print(f"  -> Saved {len(content)/1024:.1f} KB")
    except Exception as e:
        print(f"  -> Failed: {e}")

print("\n---DOWNLOADED FILES---")
for p in downloaded_paths:
    print(p)
