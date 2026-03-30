import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = r"C:\Users\dhl\data\thesis\thesis"
QUEUE_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "transcription_queue_full.csv")
OUT_PATH = os.path.join(ROOT, "Data", "Zoning_Cases", "Processed_Data", "CSV", "scraped_votes_master_index.csv")

MASTER_URLS = [
    "https://www.austintexas.gov/council/2018/2018_master_index",
    "https://www.austintexas.gov/council/2019/2019_master_index",
    "https://www.austintexas.gov/council/2020/2020_master_index"
]

def clean_dissenters(text):
    if not text: return None
    cleaned = text.replace('Council Members', '').replace('Council Member', '').replace('Mayor Pro Tem', '')
    cleaned = cleaned.replace('Mayor', '').replace('voted nay', '').replace('and', ',').strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r' ,', ',', cleaned)
    return cleaned.strip(',. ')

def main():
    df = pd.read_csv(QUEUE_PATH)
    df['Year'] = pd.to_datetime(df['Meeting_Date']).dt.year
    df = df[df['Year'].isin([2018, 2019, 2020])]

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[ 502, 503, 504 ])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    meeting_urls = []
    print("Fetching Master Indexes to find valid meeting links...")
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

    print(f"Fetching {len(meeting_urls)} individual meeting pages...")
    # Fetch them 
    html_cache = []
    for m_url in meeting_urls:
        try:
            resp = session.get(m_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                html_cache.append(BeautifulSoup(resp.content, 'html.parser').get_text(separator=' | '))
        except Exception as e:
            pass

    results = []
    print("Parsing votes from fetched meeting HTML...")
    for idx, row in df.iterrows():
        case = str(row['CASE_NUMBER']).strip()
        
        found_block = ""
        for text in html_cache:
            if case in text:
                # Basic chunking by ' | ' or 'Item'
                blocks = text.replace('Item ', ' | ').split(' | ')
                for b in blocks:
                    if case in b:
                        found_block = b
                        break
                if found_block:
                    break
                    
        # Fallback window search
        if not found_block:
            for text in html_cache:
                idx_c = text.find(case)
                if idx_c != -1:
                    found_block = text[idx_c:idx_c+500]
                    break
                    
        v_yes, v_no, nay_mem = None, None, None
        matched = False
        
        if found_block:
            match = re.search(r'[Vv]ote\s*:\s*(\d+)\s*-\s*(\d+)([^.]*)', found_block)
            if match:
                v_yes = int(match.group(1))
                v_no = int(match.group(2))
                if v_yes <= 11 and v_no <= 11:
                    nay_mem = clean_dissenters(match.group(3)) if v_no > 0 else None
                    matched = True

        results.append({
            "CASE_NUMBER": case,
            "Meeting_Date": row['Meeting_Date'],
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
