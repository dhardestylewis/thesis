import os
import re
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures

ROOT = r"C:\Users\dhl\data\thesis\thesis"
QUEUE_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "transcription_queue_full.csv")
OUT_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "scraped_backup_pdf_links.csv")

MASTER_URLS = [
    "https://www.austintexas.gov/content/archive-council-meetings-held-2007"
] + [f"https://www.austintexas.gov/council/{y}/{y}_master_index" for y in range(2008, 2026)]

import time
import random

def fetch_html(url, session):
    try:
        # Sleep randomly to avoid IP Ban
        time.sleep(random.uniform(1.5, 4.0))
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.austintexas.gov/'
        }
        resp = session.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.content.decode('utf-8', errors='ignore')
    except Exception:
        pass
    return None

def main():
    print(f"Loading {QUEUE_PATH} for PDF Backup harvests...")
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
    print(f"Found {len(meeting_urls)} total meetings to download. Executing respectful, single-threaded linear extraction with randomized sleep delays to avoid IP Blacklisting...")
    
    for i, m_url in enumerate(meeting_urls):
        res = fetch_html(m_url, session)
        if res:
            html_cache.append(res)
        if (i+1) % 10 == 0:
            print(f"Downloaded {i+1} / {len(meeting_urls)} HTML pages securely.")

                
    print(f"Downloaded {len(html_cache)} HTML meeting agendas. Parsing PDF backup links with O(1) string slicing heuristic...")

    results = []
    
    for idx, row in df.iterrows():
        case = str(row['CASE_NUMBER']).strip()
        
        found_chunk = None
        for text in html_cache:
            if case in text:
                idx_c = text.find(case)
                if idx_c != -1:
                    chunk = text[max(0, idx_c-1000):idx_c+8000]
                    item_chunks = re.split(r'\bItem \d+\b', chunk, flags=re.IGNORECASE)
                    for ic in item_chunks:
                        if case in ic:
                            found_chunk = ic
                            break
                    if not found_chunk:
                        found_chunk = chunk
                break
        
        links_data = []
        if found_chunk:
            chunk_soup = BeautifulSoup(found_chunk, 'html.parser')
            for a in chunk_soup.find_all('a', href=True):
                title = a.get_text(strip=True)
                href = a['href'].strip()
                if not href.startswith('http'):
                    href = f"https://www.austintexas.gov{href}"
                
                # Check if it meets criteria
                clower = href.lower() + title.lower()
                if 'pdf' in clower or 'document' in clower or 'ordinance' in clower or 'report' in clower:
                    links_data.append({"title": title, "url": href})
                    
        results.append({
            "CASE_NUMBER": case,
            "Meeting_Date": row.get('Meeting_Date', None),
            "backup_links": json.dumps(links_data) if links_data else "[]",
            "document_count": len(links_data)
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False)
    success_rate = (out_df['document_count'] > 0).mean() * 100
    print(f"Extracted backup documents for {success_rate:.1f}% of target cases.")
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
