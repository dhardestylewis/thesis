import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
import time
import random
from urllib3.util.retry import Retry
import concurrent.futures

ROOT = "."
OUT_DIR = os.path.join(ROOT, "Master_Meeting_Docs")
os.makedirs(OUT_DIR, exist_ok=True)

MASTER_URLS = [
    "https://www.austintexas.gov/content/archive-council-meetings-held-2007"
] + [f"https://www.austintexas.gov/council/{y}/{y}_master_index" for y in range(2008, 2026)]

def fetch_html(url, session):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return {"url": url, "html": resp.content.decode('utf-8', errors='ignore')}
    except:
        pass
    return None

def download_master_pdf(item):
    meeting_url = item['meeting_url']
    title = str(item['title']).replace('/', '-').replace('\\', '-')
    url = item['url']
    
    date_match = re.search(r'(\d{8})', meeting_url)
    date_prefix = date_match.group(1) if date_match else "UnknownDate"
    
    safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    filename = f"{date_prefix}_{safe_title}.pdf"
    filepath = os.path.join(OUT_DIR, filename)
    
    if os.path.exists(filepath):
        return True
    
    time.sleep(random.uniform(0.5, 2.0))
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(url, stream=True, headers=headers, timeout=20)
        if resp.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=16384):
                    f.write(chunk)
            return True
    except:
        pass
    return False

def main():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    meeting_urls = []
    print("Compiling global Austin Council Meeting URLs 2007-2025...")
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
    
    print(f"Discovered {len(meeting_urls)} formal Council meetings.")
    html_cache = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_html, m_url, session): m_url for m_url in meeting_urls}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                html_cache.append(res)
                
    download_queue = []
    print(f"Scanning {len(html_cache)} meeting agendas for top-level Master Documents...")
    for page in html_cache:
        url = page['url']
        html = page['html']
        
        idx = html.find("Agenda Items - ")
        if idx != -1:
            top_html = html[:idx]
        else:
            top_html = html
            
        soup = BeautifulSoup(top_html, 'html.parser')
        for a in soup.find_all('a', href=True):
            title = a.get_text(strip=True)
            href = a['href'].strip()
            
            clower = href.lower() + title.lower()
            if ('pdf' in clower or 'document' in clower or 'transcript' in clower or 'agenda' in clower):
                t_low = title.lower()
                if any(x in t_low for x in ['agenda', 'addendum', 'changes', 'questions', 'minutes', 'transcript']):
                    full_href = f"https://www.austintexas.gov{href}" if not href.startswith('http') else href
                    download_queue.append({
                        "meeting_url": url,
                        "title": title,
                        "url": full_href
                    })
                    
    print(f"Found {len(download_queue)} Master Documents. Streaming to disk...")
    successes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_master_pdf, item): item for item in download_queue}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if future.result():
                successes += 1
            if (i+1) % 100 == 0:
                print(f"Downloaded {i+1} / {len(download_queue)}")
                
    print(f"Finished downloading {successes} Master Documents successfully.")

if __name__ == "__main__":
    main()
