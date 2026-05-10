import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import re

def scrape_edims_for_petitions():
    panel_path = r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv'
    agenda_cases_path = r'c:\Users\dhl\data\Thesis\thesis\Data\interim\council_agendas_cases.csv'
    
    print("Loading data...")
    panel = pd.read_csv(panel_path, usecols=['case_number', 'petition_event'])
    protested_cases = panel[panel['petition_event'] == 1]['case_number'].unique()
    print(f"Found {len(protested_cases)} petitioned cases.")
    
    agendas = pd.read_csv(agenda_cases_path)
    target_agendas = agendas[agendas['Case_Number'].isin(protested_cases)]
    print(f"Found {len(target_agendas)} agenda mappings for these cases.")
    
    # We want to iterate through the unique URLs to minimize requests
    unique_urls = target_agendas['Meeting_URL'].unique()
    print(f"Need to scrape {len(unique_urls)} unique agenda pages...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    extracted_edims = []
    
    for i, url in enumerate(unique_urls):
        if pd.isna(url):
            continue
            
        print(f"[{i+1}/{len(unique_urls)}] Scraping {url}")
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                print(f"  -> Error {resp.status_code}")
                continue
                
            # Split HTML by hr tags to isolate individual agenda items
            html_text = resp.text
            blocks = re.split(r'<hr[^>]*>', html_text, flags=re.IGNORECASE)
            
            # Find cases for this URL
            cases_on_agenda = target_agendas[target_agendas['Meeting_URL'] == url]['Case_Number'].unique()
            
            for case in cases_on_agenda:
                # Find blocks containing this case
                for block in blocks:
                    if re.search(case, block, re.IGNORECASE):
                        soup = BeautifulSoup(block, 'html.parser')
                        links = soup.find_all('a', href=re.compile(r'edims/document\.cfm', re.IGNORECASE))
                        for link in links:
                            edims_url = link.get('href')
                            link_text = link.get_text(strip=True)
                            # Ignore empty links (like the icon images)
                            if not link_text:
                                continue
                            if not edims_url.startswith('http'):
                                edims_url = 'https://services.austintexas.gov' + edims_url
                            
                            extracted_edims.append({
                                'Case_Number': case,
                                'Meeting_URL': url,
                                'EDIMS_URL': edims_url,
                                'Link_Text': link_text
                            })
                            
            time.sleep(1) # Polite delay
            
        except Exception as e:
            print(f"  -> Exception: {e}")
            
    df_edims = pd.DataFrame(extracted_edims).drop_duplicates()
    out_path = 'Scratch/edims_petition_links.csv'
    df_edims.to_csv(out_path, index=False)
    print(f"Done! Saved {len(df_edims)} EDIMS links to {out_path}")

if __name__ == '__main__':
    scrape_edims_for_petitions()
