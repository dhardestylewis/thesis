import pandas as pd
import urllib.request
import re
import random
import time

df = pd.read_csv('c:/Users/dhl/data/Thesis/thesis/Data/austin_council_meetings_index.csv')
hpc_df = df[df['Meeting_Text'].str.contains('Housing and Planning', case=False, na=False)]

# Select 3 random meetings
sample_urls = hpc_df['URL'].sample(3).tolist()
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

print(f"Sampling 3 random Housing and Planning Committee meetings...\n")

minutes_found = []

for url in sample_urls:
    print(f"\nChecking meeting: {url}")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            docs = list(set(re.findall(r'document\.cfm\?id=(\d+)', html)))
            
        print(f"  -> Found {len(docs)} documents. Scanning headers for 'Minutes'...")
        time.sleep(1)
        
        for doc_id in docs:
            doc_url = f"https://services.austintexas.gov/edims/document.cfm?id={doc_id}"
            try:
                # Do a GET but only read headers
                req_doc = urllib.request.Request(doc_url, headers=headers)
                with urllib.request.urlopen(req_doc, timeout=15) as response_doc:
                    cd = response_doc.headers.get('Content-Disposition')
                    if cd:
                        fname_match = re.search(r'filename="([^"]+)"', cd)
                        if fname_match:
                            filename = fname_match.group(1)
                            if 'minute' in filename.lower():
                                print(f"    [MINUTES FOUND] {filename}")
                                minutes_found.append((doc_url, filename))
            except Exception as e:
                pass
            time.sleep(0.5)
            
    except Exception as e:
        print(f"  -> Failed to load meeting: {e}")

print(f"\n--- RESULTS ---")
print(f"Total minutes documents found in sample: {len(minutes_found)}")
for link, name in minutes_found:
    print(f"{name}: {link}")
