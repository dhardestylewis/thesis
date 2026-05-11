"""
scrape_pre2009_archive.py
Scrape the pre-2009 Planning Commission archive at:
https://www.austintexas.gov/boards-commissions/meetings/{YEAR}_40_archives_year_index
Go back to 2000 (oldest missing-zone cases with meaningful volume).
Extract meeting page links, then backup doc links from each meeting page.
"""
import urllib.request, re, time, os, concurrent.futures
from bs4 import BeautifulSoup
import pandas as pd

BASE = 'https://www.austintexas.gov'
OUT_DIR = r'c:\Users\dhl\data\Thesis\thesis\Data\raw\indices'
PDF_DIR = r'c:\Users\dhl\data\Thesis\thesis\Data\Pre2009_PDFs'
os.makedirs(PDF_DIR, exist_ok=True)

# Years to scrape — we have cases back to 2000 with significant volume
YEARS = list(range(2000, 2009))  # 2000-2008

def fetch_html(url, retries=2):
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read()
        except Exception as e:
            time.sleep(1)
    return None

# ── Level 1: year index pages ─────────────────────────────────────────────────
all_meeting_links = []
for year in YEARS:
    url = f"{BASE}/boards-commissions/meetings/{year}_40_archives_year_index"
    print(f"Fetching {year} index: {url}", flush=True)
    raw = fetch_html(url)
    if not raw:
        print(f"  FAILED", flush=True)
        continue
    soup = BeautifulSoup(raw, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        if 'archives' in href.lower() and str(year) in href:
            full = BASE + href if href.startswith('/') else href
            links.append({'year': year, 'text': text, 'url': full})
    print(f"  Found {len(links)} meeting links", flush=True)
    all_meeting_links.extend(links)
    time.sleep(0.3)

print(f"\nTotal meeting pages to scrape: {len(all_meeting_links)}")

# ── Level 2: meeting pages — extract backup doc links ────────────────────────
EDIMS_RE = re.compile(r'https?://services\.austintexas\.gov/edims/document\.cfm\?id=(\d+)', re.IGNORECASE)
CASE_RE  = re.compile(r'C14-\d{4}-\d{4}', re.IGNORECASE)
ZONING_KW = re.compile(r'(?:zoning|rezoning|C14|land use|NPA|backup|staff report)', re.IGNORECASE)

backup_docs = []  # list of {year, meeting_text, doc_id, doc_text, doc_url}

for mtg in all_meeting_links:
    raw = fetch_html(mtg['url'])
    if not raw:
        continue
    soup = BeautifulSoup(raw, 'html.parser')
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        # Look for EDIMS links
        m = EDIMS_RE.search(href)
        if m:
            doc_id = m.group(1)
            full_url = f"https://services.austintexas.gov/edims/document.cfm?id={doc_id}"
            backup_docs.append({
                'Year': mtg['year'],
                'Meeting_Text': mtg['text'],
                'Doc_ID': doc_id,
                'Doc_Text': text,
                'Doc_URL': full_url
            })
    # Also look for direct PDF links (pre-EDIMS format)
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.lower().endswith('.pdf'):
            full = BASE + href if href.startswith('/') else href
            text = a.get_text(strip=True)
            backup_docs.append({
                'Year': mtg['year'],
                'Meeting_Text': mtg['text'],
                'Doc_ID': None,
                'Doc_Text': text,
                'Doc_URL': full
            })
    time.sleep(0.2)

print(f"Total backup doc links found: {len(backup_docs)}")

# Save the index
df_new = pd.DataFrame(backup_docs).drop_duplicates(subset=['Doc_URL'])
out_path = os.path.join(OUT_DIR, 'pre2009_commission_index.csv')
df_new.to_csv(out_path, index=False)
print(f"Saved index: {out_path}")
print()
print("Doc type breakdown:")
def classify(t):
    t = str(t).lower()
    if 'backup' in t: return 'backup'
    if 'agenda' in t: return 'agenda'
    if 'minute' in t: return 'minutes'
    return 'other'
df_new['dtype'] = df_new['Doc_Text'].apply(classify)
print(df_new['dtype'].value_counts().to_string())
print()

# ── Download the backup docs ──────────────────────────────────────────────────
to_download = []
for _, row in df_new[df_new['dtype'] == 'backup'].iterrows():
    url = str(row['Doc_URL'])
    clean = re.sub(r'[^A-Za-z0-9_\-\. ]', '_', str(row['Doc_Text']))[:120]
    doc_id = row['Doc_ID'] or 'nodid'
    fname = f"{row['Year']}_{doc_id}_{clean}.pdf"
    fpath = os.path.join(PDF_DIR, fname)
    if not os.path.exists(fpath):
        to_download.append((url, fpath, fname))

print(f"Backup docs to download: {len(to_download)}")

def fetch_pdf(task):
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

if to_download:
    downloaded = errors = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_pdf, t): t for t in to_download}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            status, fn = fut.result()
            if status == 'ok': downloaded += 1
            else:              errors += 1
            if i % 50 == 0 or i == len(to_download):
                print(f"  [{i}/{len(to_download)}] downloaded={downloaded} errors={errors} "
                      f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    print(f"\nDone. Downloaded={downloaded} Errors={errors}")
    print(f"Saved to: {PDF_DIR}")
