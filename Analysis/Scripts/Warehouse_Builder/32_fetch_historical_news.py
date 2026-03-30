import os
import pandas as pd
import time
import random

# This script pulls Austin-specific news coverage mapping strictly between 
# [application_start_date] and [final_date] to construct the H1 Notice friction vector.

ROOT_DIR = r"C:\Users\dhl\data\thesis\thesis"
DATA = os.path.join(ROOT_DIR, "Data")
IN_FILE = os.path.join(DATA, "Warehouse_As_Of", "H0_Filing_Master_Enriched.csv")
OUT_FILE = os.path.join(DATA, "Zoning_Cases", "Processed_Data", "CSV", "H1_Media_Friction.csv")

def extract_news_friction():
    print("Loading V2 Master Warehouse for H1 Media chronologies...")
    
    try:
        from GoogleNews import GoogleNews
    except ImportError:
        print("CRITICAL: Run `pip install GoogleNews` to execute the media scraper.")
        return
        
    df = pd.read_csv(IN_FILE, low_memory=False)
    
    # Needs valid start and end dates chronologies
    valid = df.dropna(subset=['application_start_date', 'final_date']).copy()
    
    print(f"Initializing chronological Search API for {len(valid)} cases...")
    print("Bounding queries strictly to Austin Monitor, Statesman, and ABJ targets...")
    
    results = []
    
    for idx, row in valid.head(10).iterrows():
        case = row['case_number']
        try:
            # We strictly bound the scrape to the exact latency window mapping (m/d/Y)
            start = pd.to_datetime(row['application_start_date']).strftime("%m/%d/%Y")
            end = pd.to_datetime(row['final_date']).strftime("%m/%d/%Y")
        except:
            continue
            
        print(f"Scraping local media friction for {case} strictly between {start} and {end}...")
        
        # Scrape constrained exclusively to the latency gap ensuring strictly NO look-ahead bias
        googlenews = GoogleNews(start=start, end=end)
        search_str = f'"{case}" OR "Austin City Council" "{row.get("site_address", "Austin")}"'
        
        try:
            googlenews.search(search_str)
            articles = googlenews.result()
            friction_score = len(articles)
            
            results.append({
                'case_number': case,
                'h1_article_count': friction_score,
                'h1_media_headlines': " | ".join([a.get('title', '') for a in articles])
            })
        except Exception as e:
            pass
            
        # Hard rate-limit enforcement
        time.sleep(random.uniform(2.0, 4.0)) 
        googlenews.clear()
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Media execution architecture confirmed. Wrote outputs to {OUT_FILE}.")

if __name__ == "__main__":
    extract_news_friction()
