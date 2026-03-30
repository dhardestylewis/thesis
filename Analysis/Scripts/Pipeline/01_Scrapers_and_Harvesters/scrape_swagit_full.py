"""
scrape_swagit_full.py — Phase 2 (FULL SCOPE)
=============================================
Scrapes Swagit/ATXN video links for all mapped meeting agenda items.
Resolves Austin.gov meeting URLs to Swagit broadcast IDs and extracts
the pre-chunked MP4 download URLs for each specific agenda item.

Uses multithreading and per-meeting caching to avoid re-fetching
the same meeting page multiple times.
"""

import pandas as pd
import requests
import re
import os
import json
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data"
INPUT_CSV = os.path.join(DATA_DIR, "rezoning_meeting_dates_full.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "transcription_queue_full.csv")

# Cache: Austin.gov URL -> Swagit URL
swagit_cache = {}
# Cache: Swagit URL -> playlist JSON
playlist_cache = {}


def get_swagit_url(item_url):
    """Resolve an Austin.gov meeting page URL to a Swagit player URL."""
    if not isinstance(item_url, str) or not item_url.strip():
        return None
    
    # Strip fragment (e.g., #034) — we only need the base page
    base_url = item_url.split('#')[0]
    
    if base_url in swagit_cache:
        return swagit_cache[base_url]
    
    if 'swagit.com' in base_url:
        swagit_cache[base_url] = base_url
        return base_url
    
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(base_url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        for l in soup.find_all('a', href=True):
            if 'swagit.com/play/' in l['href']:
                swagit_cache[base_url] = l['href']
                return l['href']
        for i in soup.find_all('iframe', src=True):
            if 'swagit.com' in i['src']:
                swagit_cache[base_url] = i['src']
                return i['src']
    except:
        pass
    
    swagit_cache[base_url] = None
    return None


def get_playlist(swagit_url):
    """Extract the jwplayer playlist JSON from a Swagit player page."""
    if swagit_url in playlist_cache:
        return playlist_cache[swagit_url]
    
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(swagit_url, headers=headers, timeout=15)
        match = re.search(r'playlist:\s*(\[\{.*?\}\])\s*(,|})', r.text, re.DOTALL)
        if match:
            pl = json.loads(match.group(1))
            playlist_cache[swagit_url] = pl
            return pl
    except:
        pass
    
    playlist_cache[swagit_url] = []
    return []


def find_mp4(playlist, agenda_item):
    """Match an agenda item number to a playlist entry's title."""
    if not agenda_item or not playlist:
        return None
    
    agenda_str = str(agenda_item).strip()
    
    for item in playlist:
        title = item.get("title", "")
        # Exact match patterns: "Item 34", "Zoning Item 34", title == "34"
        if f"Item {agenda_str}" in title:
            return item.get("dfile")
        # Sometimes it's just the number in the title
        parts = title.split()
        if agenda_str in parts:
            return item.get("dfile")
    
    # Fuzzy: Check if the "Zoning Consent Agenda" segment exists (many items pass here)
    for item in playlist:
        title = item.get("title", "").lower()
        if "zoning" in title and ("consent" in title or "public hearing" in title):
            return item.get("dfile")
    
    return None


def worker(row_tuple):
    idx, row = row_tuple
    item_url = row.get('Item_URL', '')
    agenda_item = row.get('Agenda_Item', '')
    
    swagit_url = get_swagit_url(item_url)
    if not swagit_url:
        return None
    
    playlist = get_playlist(swagit_url)
    mp4 = find_mp4(playlist, agenda_item)
    
    if mp4:
        return {
            "CASE_NUMBER": row['CASE_NUMBER'],
            "Meeting_Date": row['Meeting_Date'],
            "Agenda_Item": agenda_item,
            "Body": row.get('Body', ''),
            "Swagit_URL": swagit_url,
            "MP4_URL": mp4
        }
    return None


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Processing {len(df)} agenda records...")
    
    results = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(worker, r): r for r in df.iterrows()}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            completed += 1
            if completed % 200 == 0:
                print(f"  [{completed}/{len(df)}] — {len(results)} MP4s found so far")
    
    out_df = pd.DataFrame(results).drop_duplicates(subset=["CASE_NUMBER", "MP4_URL"])
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nDone! Total MP4 files queued: {len(out_df)}")
    print(f"Unique cases with MP4s: {out_df['CASE_NUMBER'].nunique()}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
