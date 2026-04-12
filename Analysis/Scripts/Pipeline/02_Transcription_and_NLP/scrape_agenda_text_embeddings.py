import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

ROOT = r"C:\Users\dhl\data\thesis\thesis"
QUEUE_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "transcription_queue_full.csv")
OUT_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "scraped_agenda_text_embeddings.csv")

MASTER_URLS = [
    "https://www.austintexas.gov/content/archive-council-meetings-held-2007"
] + [f"https://www.austintexas.gov/council/{y}/{y}_master_index" for y in range(2008, 2026)]

def fetch_html(url, session):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return BeautifulSoup(resp.content, 'html.parser').get_text(separator=' | ')
    except Exception:
        pass
    return None

def main():
    print(f"Loading {QUEUE_PATH} for Text Embedding Harvesting...")
    df = pd.read_csv(QUEUE_PATH)

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    meeting_urls = []
    print("Fetching Master Indexes globally to compile all meeting links 2007-2025...")
    for url in MASTER_URLS:
        resp = session.get(url, headers=headers)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/council/20' in href and ('-reg' in href.lower() or '-spec' in href.lower() or '-spc' in href.lower()):
                    full_link = f"https://www.austintexas.gov{href}" if href.startswith('/') else href
                    if full_link not in meeting_urls:
                        meeting_urls.append(full_link)
    
    html_cache = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_html, m_url, session): m_url for m_url in meeting_urls}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                html_cache.append(res)

    print(f"Successfully downloaded {len(html_cache)} meeting agendas! Extracting raw text blocks...")
    results = []
    
    for idx, row in df.iterrows():
        case = str(row['CASE_NUMBER']).strip()
        found_block = ""
        for text in html_cache:
            if case in text:
                blocks = text.replace('Item ', ' | ').split(' | ')
                for b in blocks:
                    if case in b:
                        found_block = b
                        break
                if found_block:
                    break
                    
        if not found_block:
            for text in html_cache:
                idx_c = text.find(case)
                if idx_c != -1:
                    found_block = text[idx_c:idx_c+1000]
                    break

        if found_block:
            results.append({
                "CASE_NUMBER": case,
                "Meeting_Date": row.get('Meeting_Date', None),
                "agenda_text_raw": found_block.strip()
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out_df)} raw text blocks natively to {OUT_PATH}")

if __name__ == "__main__":
    main()
