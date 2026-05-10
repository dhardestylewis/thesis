import pandas as pd
import requests
import os
import time
import re
import fitz # PyMuPDF

def download_and_ocr():
    in_path = 'Scratch/edims_petition_links.csv'
    out_dir = r'c:\Users\dhl\data\Thesis\thesis\Data\Council_Documents\Petitions'
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(in_path):
        print(f"Missing {in_path}. Wait for scraper to finish.")
        return
        
    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} EDIMS links.")
    
    # Filter to likely petition documents
    # "petition", "protest", "backup", "staff report", "ordinance"
    keywords = 'petition|protest|backup|staff report|ordinance'
    df_filtered = df[df['Link_Text'].str.contains(keywords, case=False, na=False)].copy()
    
    # Sort to prioritize Late Backup and Staff Report
    def assign_priority(text):
        text = str(text).lower()
        if 'late backup' in text: return 1
        if 'staff report' in text: return 2
        if 'petition' in text or 'protest' in text: return 0
        return 3
        
    df_filtered['priority'] = df_filtered['Link_Text'].apply(assign_priority)
    df_filtered = df_filtered.sort_values(['Case_Number', 'priority'])
    
    results = []
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    found_cases = set()
    
    for case, group in df_filtered.groupby('Case_Number'):
        for i, row in group.iterrows():
            if case in found_cases:
                break # Skip if already found for this case
                
            url = row['EDIMS_URL']
            
            # Extract id from URL
            if 'id=' in url:
                idx = url.split('id=')[-1].split('&')[0]
            elif 'idx=' in url:
                idx = url.split('idx=')[-1].split('&')[0]
            else:
                idx = url.split('/')[-1]
            
            pdf_path = os.path.join(out_dir, f"{case}_{idx}.pdf")
            
            # Download PDF if not exists
            if not os.path.exists(pdf_path):
                print(f"Downloading {case} from {url}...")
                try:
                    resp = session.get(url, timeout=15)
                    if resp.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            f.write(resp.content)
                        time.sleep(1)
                    else:
                        print(f"  -> Failed: {resp.status_code}")
                        continue
                except Exception as e:
                    print(f"  -> Exception: {e}")
                    continue
                    
            # Extract text using PyMuPDF
            try:
                doc = fitz.open(pdf_path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text("text") + " "
                
                # Check for petition mentions
                has_petition = bool(re.search(r'valid petition|protest|petition', full_text, re.IGNORECASE))
                
                if has_petition:
                    res_path = 'Scratch/ocr_petition_results.csv'
                    res_df = pd.DataFrame([{
                        'Case_Number': case,
                        'EDIMS_URL': url,
                        'Meeting_URL': row['Meeting_URL'],
                        'Petition_Found': True
                    }])
                    res_df.to_csv(res_path, mode='a', header=not os.path.exists(res_path), index=False)
                    print(f"  *** Petition found for {case} in {row['Meeting_URL']} ***")
                    found_cases.add(case)
                
                doc.close()
            except Exception as e:
                print(f"  -> OCR Exception on {pdf_path}: {e}")
            
    print(f"Done! Process interrupted or completed. Check Scratch/ocr_petition_results.csv")

if __name__ == '__main__':
    download_and_ocr()
