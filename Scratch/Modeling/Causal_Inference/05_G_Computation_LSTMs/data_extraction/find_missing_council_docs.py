"""
find_missing_council_docs.py
The council index has 2,588 entries but council_transcripts.csv only has 376 rows.
Find what council backup docs exist in the index but haven't been downloaded yet.
Council backup documents are the most authoritative source for final approved zoning/height.
"""
import pandas as pd
import re
import os
import urllib.request
import concurrent.futures
import time

IDX_DIR      = r'c:\Users\dhl\data\Thesis\thesis\Data\raw\indices'
PDFS_DIR     = r'c:\Users\dhl\data\Thesis\thesis\Data\Council_PDFs'
COUNCIL_CSV  = r'c:\Users\dhl\data\Thesis\thesis\Data\interim\council_transcripts.csv'

os.makedirs(PDFS_DIR, exist_ok=True)

# ── What's in the council index? ─────────────────────────────────────────────
council_mtg = pd.read_csv(os.path.join(IDX_DIR, 'austin_council_meetings_index.csv'), low_memory=False)
council_min = pd.read_csv(os.path.join(IDX_DIR, 'council_minutes_index.csv'), low_memory=False)

print("=== austin_council_meetings_index.csv ===")
print(f"Shape: {council_mtg.shape}")
print(f"Cols: {list(council_mtg.columns)}")
print(council_mtg.head(5).to_string())
print()
print("Meeting_Text sample values:")
print(council_mtg['Meeting_Text'].value_counts().head(20).to_string())
print()

print("=== council_minutes_index.csv ===")
print(f"Shape: {council_min.shape}")
print(f"Cols: {list(council_min.columns)}")
print(council_min.head(5).to_string())
print()

# ── What do we already have in council_transcripts.csv? ──────────────────────
print("=== CURRENT council_transcripts.csv ===")
ct = pd.read_csv(COUNCIL_CSV, low_memory=False)
print(f"Shape: {ct.shape}")
print(f"Cols: {list(ct.columns)}")
print()

# ── Check if council_meetings_index has Doc_URLs we can fetch ─────────────────
if 'URL' in council_mtg.columns:
    urls = council_mtg['URL'].dropna()
    print(f"Council meeting URLs in index: {len(urls)}")
    print("Sample URLs:")
    print(urls.head(5).to_string())
    print()

    # Classify what types of documents these are
    def classify_url(u):
        u = str(u).lower()
        if 'backup' in u:    return 'backup'
        if 'agenda' in u:    return 'agenda'
        if 'minutes' in u:   return 'minutes'
        if 'ordinance' in u: return 'ordinance'
        return 'other'

    council_mtg['dtype'] = council_mtg['URL'].apply(classify_url)
    print("URL type breakdown:")
    print(council_mtg['dtype'].value_counts().to_string())
    print()

# ── Check which PDFs are already on disk ─────────────────────────────────────
existing_pdfs = set()
if os.path.exists(PDFS_DIR):
    existing_pdfs = set(os.listdir(PDFS_DIR))
print(f"Council PDFs already on disk: {len(existing_pdfs)}")
print()

# Also check Commission_PDFs for any council docs
comm_pdfs_dir = r'c:\Users\dhl\data\Thesis\thesis\Data\Commission_PDFs'
if os.path.exists(comm_pdfs_dir):
    comm_files = os.listdir(comm_pdfs_dir)
    print(f"Commission_PDFs dir: {len(comm_files)} files")
    # Sample some names to see naming convention
    print("Sample filenames:")
    for f in sorted(comm_files)[:5]:
        print(f"  {f}")
    print()

# ── Build download list from council_meetings_index ───────────────────────────
# The meetings index has Year, Meeting_Text, URL
# Meeting_Text describes what the doc is
if 'URL' in council_mtg.columns and 'Meeting_Text' in council_mtg.columns:
    to_download = []
    for _, row in council_mtg.iterrows():
        url = str(row.get('URL', ''))
        if not url or url == 'nan' or not url.startswith('http'):
            continue
        year = str(row.get('Year', 'unk'))
        text = re.sub(r'[^A-Za-z0-9_\-\. ]', '_', str(row.get('Meeting_Text', 'doc')))[:120]
        # Use URL hash as ID since no Doc_ID column
        import hashlib
        doc_id = hashlib.md5(url.encode()).hexdigest()[:8]
        fname = f"{year}_{doc_id}_{text}.pdf"
        fpath = os.path.join(PDFS_DIR, fname)
        if fname not in existing_pdfs and not os.path.exists(fpath):
            to_download.append((url, fpath, fname))

    print(f"Council docs to download: {len(to_download)}")

    if len(to_download) > 0:
        print("Sample download targets:")
        for url, fp, fn in to_download[:10]:
            print(f"  {fn[:80]}")
            print(f"    URL: {url[:80]}")
        print()

        # Download
        downloaded = errors = skipped = 0

        def fetch(task):
            url, path, fn = task
            if os.path.exists(path):
                return ('skip', fn)
            for attempt in range(2):
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=25) as r:
                        content = r.read()
                        # Only save if it looks like a PDF
                        if len(content) > 1000:
                            with open(path, 'wb') as f:
                                f.write(content)
                            return ('ok', fn)
                        else:
                            return ('empty', fn)
                except Exception:
                    time.sleep(1)
            return ('err', fn)

        t0 = time.time()
        print(f"Starting download with 15 workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(fetch, t): t for t in to_download}
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                status, fn = fut.result()
                if status == 'ok':   downloaded += 1
                elif status == 'skip': skipped += 1
                else:                errors += 1
                if i % 100 == 0 or i == len(to_download):
                    elapsed = time.time() - t0
                    print(f"  [{i}/{len(to_download)}] "
                          f"downloaded={downloaded} skipped={skipped} errors={errors} "
                          f"rate={i/elapsed:.1f}/s  elapsed={elapsed/60:.1f}min")

        print(f"\nDone. Downloaded={downloaded} Errors={errors} "
              f"Time={(time.time()-t0)/60:.1f}min")
        print(f"Council PDFs saved to: {PDFS_DIR}")
    else:
        print("Nothing to download — all council docs already on disk.")
