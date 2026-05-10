"""
Phase 3: Download LDB 2016 and 2021 from CoA Open Data portal.
These are the TCAD parcel-level appraisal databases required for the enrichment.
"""
import requests, os, time

ROOT = r'C:\Users\dhl\data\thesis\thesis'
COA_DIR = os.path.join(ROOT, 'Data', 'CoA_Open_Data')
os.makedirs(COA_DIR, exist_ok=True)

# Socrata download URLs from the MANIFEST dataset IDs
DOWNLOADS = [
    ('LDB_2016_4nsn-uea6.csv', 'https://data.austintexas.gov/api/views/4nsn-uea6/rows.csv?accessType=DOWNLOAD'),
    ('LDB_2021_kk8y-6cmt.csv', 'https://data.austintexas.gov/api/views/kk8y-6cmt/rows.csv?accessType=DOWNLOAD'),
]

for filename, url in DOWNLOADS:
    out_path = os.path.join(COA_DIR, filename)

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[*] {filename} already exists ({size_mb:.0f} MB), skipping.")
        continue

    print(f"[*] Downloading {filename}...")
    print(f"    URL: {url}")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 200:
            total = 0
            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
                    total += len(chunk)
                    if total % (50*1024*1024) == 0:
                        print(f"    Downloaded {total/1e6:.0f} MB...")
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"    Done. {size_mb:.1f} MB written to {out_path}")
        else:
            print(f"    ERROR: HTTP {resp.status_code}")
    except Exception as e:
        print(f"    ERROR: {e}")

print("\n[*] LDB download phase complete.")
