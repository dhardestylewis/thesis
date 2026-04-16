import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = r"C:\Users\dhl\data\thesis\thesis"
QUEUE_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "transcription_queue_full.csv")
OUT_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "scraped_council_votes.csv")

def build_url(date_str):
    if pd.isna(date_str):
        return None
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        yyyy = dt.strftime("%Y")
        yyyymmdd = dt.strftime("%Y%m%d")
        return f"https://www.austintexas.gov/council/{yyyy}/{yyyymmdd}-reg"
    except Exception as e:
        return None

def main():
    print(f"Loading {QUEUE_PATH}...")
    df = pd.read_csv(QUEUE_PATH)
    
    unique_dates = df['Meeting_Date'].dropna().unique()
    html_cache = {}
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    print(f"Fetching {len(unique_dates)} distinct meeting HTML pages...")
    for date_str in unique_dates:
        url = build_url(date_str)
        if url:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                resp = session.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    text = BeautifulSoup(resp.content, 'html.parser').get_text(separator=' ')
                    html_cache[date_str] = text
                else:
                    html_cache[date_str] = None
            except Exception as e:
                html_cache[date_str] = None
    
    results = []
    print("Parsing votes from HTML text...")
    for idx, row in df.iterrows():
        case = str(row['CASE_NUMBER']).strip()
        date_str = row['Meeting_Date']
        
        text = html_cache.get(date_str)
        if not text or pd.isna(row['CASE_NUMBER']):
            results.append({"CASE_NUMBER": case, "Meeting_Date": date_str, "vote_yes": None, "vote_no": None, "nay_members": None, "matched": False})
            continue
            
        items = re.split(r'\bItem\s+\d+\b', text, flags=re.IGNORECASE)
        found_block = ""
        for item in items:
            if case in item:
                found_block = item
                break
        
        if not found_block:
            idx_c = text.find(case)
            if idx_c != -1:
                found_block = text[max(0, idx_c-500):idx_c+2000]
        
        matched = False
        v_yes, v_no, nay_mem = None, None, None
        
        if found_block:
            # Capture Vote: 11-0 or Vote: 9-3, Council...
            match = re.search(r'[vV]ote[\s:]+(\d+)\s*-\s*(\d+)([^.]*)', found_block)
            if match:
                v_yes = int(match.group(1))
                v_no = int(match.group(2))
                if v_yes <= 11 and v_no <= 11:
                    nay_mem_text = match.group(3).strip()
                    nay_mem_text = nay_mem_text.replace(',', '').replace('Council Members', '').replace('Council Member', '').replace('voted nay', '').strip()
                    nay_mem = nay_mem_text if v_no > 0 and len(nay_mem_text) > 0 else None
                    matched = True

        results.append({
            "CASE_NUMBER": case,
            "Meeting_Date": date_str,
            "vote_yes": v_yes,
            "vote_no": v_no,
            "nay_members": nay_mem,
            "matched": matched
        })
    
    out_df = pd.DataFrame(results)
    success_rate = out_df['matched'].mean() * 100
    print(f"Extracted {out_df['matched'].sum()} / {len(out_df)} votes successfully ({success_rate:.1f}%).")
    
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
