import os
import json
import pandas as pd
import requests
import concurrent.futures
import time
import random

ROOT = "."
PDF_DIR = os.path.join(ROOT, "PDFs")
CSV_PATH = os.path.join(ROOT, "scraped_backup_pdf_links.csv")

os.makedirs(PDF_DIR, exist_ok=True)

def download_pdf(item):
    case_num = item['case_num']
    title = str(item['title']).replace('/', '-').replace('\\', '-')
    url = item['url']
    
    safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    filename = f"{case_num}_{safe_title}.pdf"
    filepath = os.path.join(PDF_DIR, filename)
    
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
    except Exception as e:
        print(f"FAILED on {url} - Error: {e}")
    return False

def main():
    df = pd.read_csv(CSV_PATH)
    
    download_queue = []
    for idx, row in df.iterrows():
        case = row['CASE_NUMBER']
        links_str = row['backup_links']
        
        if pd.isna(links_str) or links_str == '[]':
            continue
            
        try:
            links = json.loads(links_str)
            for l in links:
                t = str(l['title']).lower()
                # We want pre-vote NLP signals, NOT post-vote Ordinances which leak target variables!
                # We also want to skip useless visual 'Exhibits'
                if any(x in t for x in ['staff report', 'petition', 'late backup', 'recommendation', 'presentation', 'response']):
                    download_queue.append({
                        "case_num": case,
                        "title": l['title'],
                        "url": l['url']
                    })
        except:
            pass
            
    print(f"Starting highly parallelized download of {len(download_queue)} PDF documents...")
    
    successes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_pdf, item): item for item in download_queue}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if future.result():
                successes += 1
            if (i+1) % 500 == 0:
                print(f"Downloaded {i+1} / {len(download_queue)} PDFs...")
                
    print(f"\nSuccessfully downloaded {successes} / {len(download_queue)} PDFs to {PDF_DIR}")

if __name__ == "__main__":
    main()
