import pandas as pd
import requests
import re
import os
import json
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data"
INPUT_CSV = os.path.join(DATA_DIR, "rezoning_meeting_dates.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "transcription_queue.csv")

def get_swagit_url(item_url):
    if not isinstance(item_url, str): return None
    if 'swagit.com' in item_url:
        return item_url
    
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(item_url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        for l in soup.find_all('a', href=True):
            if 'swagit.com/play/' in l['href']:
                return l['href']
        for i in soup.find_all('iframe', src=True):
            if 'swagit.com' in i['src']:
                return i['src']
    except:
        pass
    return None

def parse_swagit_playlist(swagit_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(swagit_url, headers=headers, timeout=10)
        # Look for the jwplayer playlist json
        match = re.search(r'playlist:\s*(\[\{.*?\}\])\s*(,|})', r.text, re.DOTALL | re.IGNORECASE)
        if match:
            return json.loads(match.group(1))
    except:
        pass
    return []

def worker(row_tuple):
    idx, row = row_tuple
    case = row['CASE_NUMBER']
    agenda_num = str(row['Agenda_Item'])
    item_url = row['Item_URL']
    
    swagit_url = get_swagit_url(item_url)
    if not swagit_url:
        return None
        
    playlist = parse_swagit_playlist(swagit_url)
    
    target_mp4 = None
    for item in playlist:
        title = item.get("title", "")
        # Try to match the item number
        if f"Item {agenda_num}" in title or agenda_num in title.split():
            target_mp4 = item.get("dfile")
            break
            
    if target_mp4:
        print(f"MATCH: {case} -> {target_mp4}")
        return {
            "CASE_NUMBER": case,
            "Meeting_Date": row["Meeting_Date"],
            "Agenda_Item": agenda_num,
            "Swagit_URL": swagit_url,
            "MP4_URL": target_mp4
        }
    return None

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Processing {len(df)} meeting records...")
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(worker, r): r for r in df.iterrows()}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                
    out_df = pd.DataFrame(results).drop_duplicates()
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Queued {len(out_df)} MP4 files for transcription.")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
