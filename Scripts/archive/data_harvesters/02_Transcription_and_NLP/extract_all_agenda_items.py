"""
extract_all_agenda_items.py — Phase 1 (FULL SCOPE)
===================================================
Maps ALL 6,865 zoning cases from edir-dcnf to their City Council,
Planning Commission, and ZAP Commission meeting dates and agenda items.

Strategy:
  1. Historical Agenda (akgy-tbxy, 2004-2020): Has a dedicated
     `zoning_case_number` column — download the ENTIRE dataset once
     and join locally (fast, no per-case API calls).
  2. Council Items Updates 2015-2024 (wsf2-3rpw): Full-text search
     on `description` for each case number pattern.
  3. Council Items Updates 2024-Present (sich-49ay): Full-text $q search.

Author: Daniel Hardesty Lewis (auto-generated pipeline)
"""

import pandas as pd
import requests
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
ZC_CSV = r"C:\Users\dhl\data\thesis\thesis\Data\CoA_Open_Data\Zoning\ZC_current_edir-dcnf.csv"
DATA_DIR = r"C:\Users\dhl\data\thesis\thesis\Data\Zoning_Cases\Processed_Data"
OUTPUT_CSV = os.path.join(DATA_DIR, "rezoning_meeting_dates_full.csv")
HISTORICAL_CACHE = os.path.join(DATA_DIR, "historical_agenda_cache.csv")


def download_historical_agenda():
    """Download the entire Historical Agenda dataset (2004-2020) once."""
    if os.path.exists(HISTORICAL_CACHE):
        print(f"  Using cached historical agenda: {HISTORICAL_CACHE}")
        return pd.read_csv(HISTORICAL_CACHE)

    print("  Downloading full Historical Agenda dataset (akgy-tbxy)...")
    url = "https://data.austintexas.gov/resource/akgy-tbxy.csv"
    all_rows = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "$limit": page_size,
            "$offset": offset,
            "$order": ":id",
            "$select": "zoning_case_number,meeting_date,meeting_type,body,agenda_item_number,item_description,agenda_item_number_id,link_to_clerks_website"
        }
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=60)
                r.raise_for_status()
                break
            except Exception as e:
                print(f"    Retry {attempt+1}/3 at offset={offset}: {e}")
                time.sleep(2)
        else:
            print(f"    FAILED at offset={offset}, skipping.")
            break

        lines = r.text.strip().split('\n')
        if len(lines) <= 1 and offset > 0:
            break
        if offset == 0:
            all_rows.append(r.text)
        else:
            all_rows.append('\n'.join(lines[1:]))  # skip header
        row_count = len(lines) - 1
        print(f"    offset={offset}, rows={row_count}")
        if row_count < page_size:
            break
        offset += page_size
        time.sleep(0.5)

    combined = '\n'.join(all_rows)
    with open(HISTORICAL_CACHE, 'w', encoding='utf-8') as f:
        f.write(combined)

    return pd.read_csv(HISTORICAL_CACHE)


def search_updates_2015(case_number):
    """Search wsf2-3rpw for a case number in description field."""
    url = "https://data.austintexas.gov/resource/wsf2-3rpw.json"
    params = {"$where": f"description like '%{case_number}%'", "$limit": 50}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []


def search_updates_2024(case_number):
    """Search sich-49ay using full-text $q."""
    url = f"https://data.austintexas.gov/resource/sich-49ay.json?$q={case_number}&$limit=50"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []


def process_case_updates(case_number):
    """Query the two update datasets for a single case number."""
    results = []

    # 2015-2024 Updates
    for row in search_updates_2015(case_number):
        results.append({
            "CASE_NUMBER": case_number,
            "Meeting_Date": row.get("agenda_date"),
            "Agenda_Item": row.get("item_number"),
            "Body": "City Council",
            "Description": row.get("description", "")[:200],
            "Source": "wsf2-3rpw",
            "Item_URL": row.get("item_url", {}).get("url", "") if isinstance(row.get("item_url"), dict) else ""
        })

    # 2024-Present Updates
    for row in search_updates_2024(case_number):
        results.append({
            "CASE_NUMBER": case_number,
            "Meeting_Date": row.get("agenda_date"),
            "Agenda_Item": row.get("item_number"),
            "Body": "City Council",
            "Description": row.get("posting_language", "")[:200],
            "Source": "sich-49ay",
            "Item_URL": row.get("attachments", {}).get("url", "") if isinstance(row.get("attachments"), dict) else ""
        })

    return results


def main():
    print("=" * 60)
    print("Phase 1 (FULL SCOPE): Map ALL zoning cases to meetings")
    print("=" * 60)

    # Load all case numbers
    zc_df = pd.read_csv(ZC_CSV)
    all_cases = zc_df['CASE_NUMBER'].dropna().unique()
    print(f"\nTotal unique zoning cases: {len(all_cases)}")

    # ── Step 1: Historical Agenda (bulk local join) ──
    print("\n[Step 1] Historical Agenda Items (2004-2020)...")
    hist_df = download_historical_agenda()
    print(f"  Historical agenda rows: {len(hist_df)}")

    # Filter to rows that have a zoning_case_number
    hist_zc = hist_df[hist_df['zoning_case_number'].notna()].copy()
    print(f"  Rows with zoning_case_number: {len(hist_zc)}")

    # Join: find all historical agenda items for our cases
    case_set = set(all_cases)
    hist_matches = hist_zc[hist_zc['zoning_case_number'].isin(case_set)]
    print(f"  Matched to our cases: {len(hist_matches)}")

    results = []
    for _, row in hist_matches.iterrows():
        link = row.get('link_to_clerks_website', '')
        if isinstance(link, str) and link.startswith('{'):
            # Sometimes stored as dict-like string
            try:
                import ast
                link = ast.literal_eval(link).get('url', '')
            except:
                pass
        results.append({
            "CASE_NUMBER": row['zoning_case_number'],
            "Meeting_Date": row.get('meeting_date'),
            "Agenda_Item": row.get('agenda_item_number'),
            "Body": row.get('body', 'City Council'),
            "Description": str(row.get('item_description', ''))[:200],
            "Source": "akgy-tbxy",
            "Item_URL": link if isinstance(link, str) else ""
        })

    print(f"  Historical results: {len(results)}")

    # ── Step 2: Query update datasets (threaded) ──
    print(f"\n[Step 2] Council Items Updates (2015-Present)...")
    print(f"  Querying {len(all_cases)} cases across 2 update datasets (10 threads)...")

    update_results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_case_updates, c): c for c in all_cases}
        for future in as_completed(futures):
            res = future.result()
            if res:
                update_results.extend(res)
            completed += 1
            if completed % 200 == 0:
                print(f"    Processed {completed}/{len(all_cases)} cases, found {len(update_results)} records so far...")

    print(f"  Update results: {len(update_results)}")
    results.extend(update_results)

    # ── Combine and deduplicate ──
    out_df = pd.DataFrame(results).drop_duplicates(subset=["CASE_NUMBER", "Meeting_Date", "Agenda_Item", "Source"])
    out_df.to_csv(OUTPUT_CSV, index=False)

    unique_cases_found = out_df['CASE_NUMBER'].nunique()
    print(f"\n{'=' * 60}")
    print(f"DONE! Total agenda records: {len(out_df)}")
    print(f"Unique cases mapped: {unique_cases_found} / {len(all_cases)}")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
