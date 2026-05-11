"""
parse_council_pages.py
The 2,412 downloaded council "PDFs" are actually HTML meeting index pages.
Parse each one to extract links to individual case backup documents, then download those.
"""
import os, re, time, urllib.request, concurrent.futures
from bs4 import BeautifulSoup

PAGES_DIR = r'c:\Users\dhl\data\Thesis\thesis\Data\Council_PDFs'
BACKUP_DIR = r'c:\Users\dhl\data\Thesis\thesis\Data\Council_Backups'
os.makedirs(BACKUP_DIR, exist_ok=True)

pages = [f for f in os.listdir(PAGES_DIR) if os.path.exists(os.path.join(PAGES_DIR, f))]
print(f"Council pages to parse: {len(pages)}")

# Parse each page for backup/zoning case PDF links
backup_links = {}  # url -> filename
EDIMS_RE = re.compile(r'https?://services\.austintexas\.gov/edims/document\.cfm\?id=(\d+)', re.IGNORECASE)
CASE_RE  = re.compile(r'C14-\d{4}-\d{4}', re.IGNORECASE)

zoning_keywords = re.compile(
    r'(?:zoning|rezoning|C14-|land use|neighborhood plan|NPA-|compatibility)',
    re.IGNORECASE
)

n_parsed = n_links = n_zoning = 0
for fname in pages:
    fpath = os.path.join(PAGES_DIR, fname)
    try:
        with open(fpath, 'rb') as f:
            raw = f.read()
        # Check if it's HTML (not PDF)
        if raw[:4] == b'%PDF':
            continue
        soup = BeautifulSoup(raw, 'html.parser')
        n_parsed += 1
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            # Look for EDIMS document links
            m = EDIMS_RE.search(href)
            if m:
                doc_id = m.group(1)
                full_url = f"https://services.austintexas.gov/edims/document.cfm?id={doc_id}"
                # Is this a zoning-related backup?
                if zoning_keywords.search(text) or CASE_RE.search(text):
                    clean = re.sub(r'[^A-Za-z0-9_\-\. ]', '_', text)[:100]
                    out_fname = f"{doc_id}_{clean}.pdf"
                    if out_fname not in backup_links:
                        backup_links[full_url] = out_fname
                        n_zoning += 1
                else:
                    n_links += 1
    except Exception:
        continue

print(f"Parsed {n_parsed} HTML pages")
print(f"Total EDIMS links found: {n_links + n_zoning}")
print(f"Zoning-related backup links: {n_zoning}")
print()

# Filter to only those not already downloaded
already = set(os.listdir(BACKUP_DIR))
to_fetch = [(url, os.path.join(BACKUP_DIR, fn), fn)
            for url, fn in backup_links.items()
            if fn not in already]
print(f"New zoning backup docs to download: {len(to_fetch)}")

if len(to_fetch) == 0:
    print("Nothing to fetch.")
else:
    print("Sample targets:")
    for url, fp, fn in to_fetch[:10]:
        print(f"  {fn[:80]}")

    def fetch(task):
        url, path, fn = task
        for _ in range(2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=25) as r:
                    content = r.read()
                if len(content) > 500:
                    with open(path, 'wb') as f:
                        f.write(content)
                    return ('ok', fn)
                return ('empty', fn)
            except Exception:
                time.sleep(1)
        return ('err', fn)

    downloaded = errors = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        futs = {ex.submit(fetch, t): t for t in to_fetch}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            status, fn = fut.result()
            if status == 'ok': downloaded += 1
            else:              errors += 1
            if i % 100 == 0 or i == len(to_fetch):
                print(f"  [{i}/{len(to_fetch)}] downloaded={downloaded} errors={errors} "
                      f"rate={i/(time.time()-t0):.1f}/s  elapsed={(time.time()-t0)/60:.1f}min")

    print(f"\nDone. Downloaded={downloaded} Errors={errors} Time={(time.time()-t0)/60:.1f}min")
    print(f"Saved to: {BACKUP_DIR}")
