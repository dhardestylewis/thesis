import pandas as pd
import requests
import time
import os

# Paths
DATA_DIR = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data"
INPUT_CSV = os.path.join(DATA_DIR, "multi_parcel_closed_2018_2025.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "rezoning_meeting_dates.csv")

def search_soda(dataset_id, where_clause):
    url = f"https://data.austintexas.gov/resource/{dataset_id}.json"
    params = {
        "$where": where_clause,
        "$limit": 50
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Error querying {dataset_id}: {e}")
    return []

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    cases = df['CASE_NUMBER'].dropna().unique()
    print(f"Found {len(cases)} unique zoning cases.")
    
    results = []
    
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Querying meeting dates for {case}...")
        
        # 1. Historical Data (2004-2020) - akgy-tbxy
        res1 = search_soda("akgy-tbxy", f"zoning_case_number like '%{case}%'")
        for row in res1:
            results.append({
                "CASE_NUMBER": case,
                "Meeting_Date": row.get("meeting_date"),
                "Agenda_Item": row.get("agenda_item_number"),
                "Body": row.get("body", "City Council"),
                "Description": row.get("item_description", ""),
                "Source": "akgy-tbxy",
                "Item_URL": row.get("link_to_clerks_website", {}).get("url", "")
            })
            
        # 2. Updates (2015-2024) - wsf2-3rpw
        res2 = search_soda("wsf2-3rpw", f"description like '%{case}%'")
        for row in res2:
            results.append({
                "CASE_NUMBER": case,
                "Meeting_Date": row.get("agenda_date"),
                "Agenda_Item": row.get("item_number"),
                "Body": "City Council", # CIUR is always Council
                "Description": row.get("description", ""),
                "Source": "wsf2-3rpw",
                "Item_URL": row.get("item_url", {}).get("url", "")
            })
            
        # 3. Updates (2024-Present) - sich-49ay
        # Note: 'attachments' is a URL column so we can't 'like' it. We must search 'tags' or we just 'like' across all text if possible, but let's try searching 'item_type' or just skip for now, since 2018-2025 is mostly covered by first two or we use full text search $q
        url3 = f"https://data.austintexas.gov/resource/sich-49ay.json?$q={case}"
        try:
            r3 = requests.get(url3, timeout=10)
            if r3.status_code == 200:
                for row in r3.json():
                    results.append({
                        "CASE_NUMBER": case,
                        "Meeting_Date": row.get("agenda_date"),
                        "Agenda_Item": row.get("item_number"),
                        "Body": "City Council",
                        "Description": row.get("tags", ""),
                        "Source": "sich-49ay",
                        "Item_URL": row.get("attachments", {}).get("url", "")
                    })
        except:
            pass
            
        time.sleep(0.1) # Be a good citizen
        
    out_df = pd.DataFrame(results).drop_duplicates()
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Found {len(out_df)} meeting agenda records.")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
