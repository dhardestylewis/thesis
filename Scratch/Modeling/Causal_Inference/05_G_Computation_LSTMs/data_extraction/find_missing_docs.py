"""
find_missing_docs.py
Compare the full commission/ZAP index against what's already in commission_transcripts.csv.
Download only the delta — docs in the index but not yet transcribed.
Also check for a council backup doc index (separate from vote transcripts).
"""
import pandas as pd
import re
import os
import urllib.request
import concurrent.futures
import time

IDX_DIR   = r'c:\Users\dhl\data\Thesis\thesis\Data\raw\indices'
PDFS_DIR  = r'c:\Users\dhl\data\Thesis\thesis\Data\Commission_PDFs'
TRANS_CSV = r'c:\Users\dhl\data\Thesis\thesis\Data\interim\commission_transcripts.csv'

os.makedirs(PDFS_DIR, exist_ok=True)

# Load the full index
plan = pd.read_csv(os.path.join(IDX_DIR, 'planning_commission_index.csv'), low_memory=False)
zap  = pd.read_csv(os.path.join(IDX_DIR, 'zoning_platting_commission_index.csv'), low_memory=False)
idx  = pd.concat([plan, zap], ignore_index=True).drop_duplicates(subset=['Doc_ID'])
print(f"Full combined index: {len(idx)} unique Doc_IDs")

# What's already in our transcripts corpus?
trans = pd.read_csv(TRANS_CSV, low_memory=False, usecols=['Filename'])
# Doc_ID is embedded in the filename: "YEAR_DOCID_DocText.pdf"
def extract_doc_id(fn):
    m = re.match(r'^\d{4}_(\d+)_', str(fn))
    return int(m.group(1)) if m else None

trans['Doc_ID'] = trans['Filename'].apply(extract_doc_id)
already_have = set(trans['Doc_ID'].dropna().astype(int).tolist())
print(f"Already transcribed Doc_IDs: {len(already_have)}")

# Delta
idx['Doc_ID_int'] = pd.to_numeric(idx['Doc_ID'], errors='coerce')
missing = idx[~idx['Doc_ID_int'].isin(already_have)].copy()
print(f"Missing (in index but not transcribed): {len(missing)}")
print()

if len(missing) == 0:
    print("NOTHING TO DOWNLOAD — corpus is complete against the current index.")
else:
    # Classify missing docs
    def classify(t):
        t = str(t).lower()
        if 'backup' in t: return 'backup'
        if 'agenda' in t: return 'agenda'
        if 'minute' in t: return 'minutes'
        return 'other'

    missing['dtype'] = missing['Doc_Text'].apply(classify)
    print("Missing doc types:")
    print(missing['dtype'].value_counts().to_string())
    print()

    # Focus on backup docs only
    missing_backup = missing[missing['dtype'] == 'backup'].copy()
    print(f"Missing backup/staff report docs: {len(missing_backup)}")
    if len(missing_backup) > 0:
        print(missing_backup[['Year','Meeting_Date','Doc_ID','Doc_Text','Doc_URL']].head(20).to_string(index=False))
        print()

        # Download them
        def make_filename(row):
            clean = re.sub(r'[^A-Za-z0-9_\-\. ]', '_', str(row['Doc_Text']))[:150]
            return f"{row['Year']}_{row['Doc_ID']}_{clean}.pdf"

        tasks = []
        for _, row in missing_backup.iterrows():
            fn  = make_filename(row)
            fp  = os.path.join(PDFS_DIR, fn)
            url = str(row['Doc_URL'])
            if not os.path.exists(fp):
                tasks.append((url, fp, fn))

        print(f"PDFs to download (not on disk): {len(tasks)}")

        if len(tasks) > 0:
            downloaded = errors = 0

            def fetch(task):
                url, path, fn = task
                for _ in range(2):
                    try:
                        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=20) as r:
                            with open(path, 'wb') as f:
                                f.write(r.read())
                        return ('ok', fn)
                    except Exception as e:
                        time.sleep(1)
                return ('err', fn)

            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
                futs = {ex.submit(fetch, t): t for t in tasks}
                for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                    status, fn = fut.result()
                    if status == 'ok': downloaded += 1
                    else:              errors += 1
                    if i % 50 == 0 or i == len(tasks):
                        print(f"  [{i}/{len(tasks)}] downloaded={downloaded} errors={errors} "
                              f"rate={i/(time.time()-t0):.1f}/s")

            print(f"\nDone. Downloaded: {downloaded}  Errors: {errors}  "
                  f"Time: {(time.time()-t0)/60:.1f} min")
        else:
            print("All missing backup PDFs already on disk (just not yet transcribed).")

# Also check for a council-specific backup index
print()
print("=== CHECKING FOR COUNCIL BACKUP DOC INDEX ===")
for fname in os.listdir(IDX_DIR):
    if 'council' in fname.lower():
        df = pd.read_csv(os.path.join(IDX_DIR, fname), low_memory=False)
        print(f"{fname}: {df.shape}  cols={list(df.columns)}")
        if 'Doc_Text' in df.columns:
            def ct(t):
                t = str(t).lower()
                if 'backup' in t: return 'backup'
                if 'agenda' in t: return 'agenda'
                if 'minute' in t: return 'minutes'
                return 'other'
            print(df['Doc_Text'].apply(ct).value_counts().to_string())
        print()
