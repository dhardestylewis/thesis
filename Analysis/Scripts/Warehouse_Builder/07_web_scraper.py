import os
import json
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
WORK_DIR = os.path.join(ROOT_DIR, "Data", "Warehouse_As_Of", "Build")
OUT_DIR = os.path.join(ROOT_DIR, "Data", "Scraped_Agendas")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_FILE = os.path.join(OUT_DIR, "scraper_cache.json")
BASE_URL = "https://www.austintexas.gov"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
else:
    cache = {}

def get_html(url, session):
    if url in cache:
        print(f"CACHE HIT: {url}")
        return cache[url]
    
    print(f"FETCHING: {url}")
    time.sleep(2.5)  # Defensive rate limiting
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        html = response.text
        cache[url] = html
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        return html
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def orchestrate_scraper():
    print("Initiating Austin City Clerk Phase 1 Agenda Scraper...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8'
    })
    
    meeting_records = []
    
    for year in range(2000, 2025): # Scrape 2000 to 2024
        if year in [2000, 2001, 2002, 2003]:
            path = f"/council/archive/{year}_council_index"
        elif year in [2006, 2007]:
            path = f"/content/archive-council-meetings-held-{year}"
        else:
            path = f"/council/{year}/{year}_master_index"
            
        index_url = BASE_URL + path
        html = get_html(index_url, session)
        
        if not html:
            continue
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for <a class="edims" href="..."> or fallback to any <a> tag mapping /council/...
        links = soup.find_all('a')
            
        valid_meetings = 0
        for link in links:
            href = link.get('href', '')
            if href and "/council/" in href and "-" in href and (len(href.split('/')[-1]) > 8):
                title = link.get_text(strip=True)
                if not title:
                    title = "Unknown Session"
                    
                full_url = BASE_URL + href if href.startswith('/') else href
                # Exclude duplicate base indexes
                if "master_index" not in full_url:
                    meeting_records.append({
                        'year': year,
                        'meeting_title': title,
                        'meeting_url': full_url
                    })
                    valid_meetings += 1
                
        print(f"Year {year}: Scraped {valid_meetings} agenda targets.")

    # Export
    out_csv = os.path.join(OUT_DIR, "meeting_index_master.csv")
    df = pd.DataFrame(meeting_records)
    
    # Deduplicate strictly on the URL
    if not df.empty:
        df = df.drop_duplicates(subset=['meeting_url'])
        df.to_csv(out_csv, index=False)
        print(f"\nPhase 1 Spidering Complete. {len(df)} discrete Austin City Council meetings natively indexed to:")
        print(out_csv)
    else:
        print("\nPhase 1 Failed. No meeting targets were cleanly resolved from the DOM.")

if __name__ == "__main__":
    orchestrate_scraper()
