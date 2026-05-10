"""
Phase 2: Geocode lat/lon → Census tract GEOID using Census Geocoder batch API.
Chunks cases into batches of 1000. No API key needed.
Output: Data/Panel/case_geoid_lookup.csv
"""
import pandas as pd, requests, io, os, time

ROOT = r'C:\Users\dhl\data\thesis\thesis'
OUT_PATH = os.path.join(ROOT, 'Data', 'Panel', 'case_geoid_lookup.csv')
GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"

sc = pd.read_parquet(os.path.join(ROOT, 'Data', 'interim', 'stage_c_features_raw.parquet'))
has_ll = sc['latitude'].notna() & sc['longitude'].notna()
cases = sc[has_ll][['case_id', 'latitude', 'longitude']].copy().reset_index(drop=True)
print(f"[*] Geocoding {len(cases)} cases with lat/lon...")

results = []
BATCH = 1000

for start in range(0, len(cases), BATCH):
    batch = cases.iloc[start:start+BATCH]
    batch_num = start // BATCH + 1
    total_batches = (len(cases) + BATCH - 1) // BATCH
    print(f"  Batch {batch_num}/{total_batches} ({len(batch)} cases)...")

    # Build address file for batch geocoder (id, street, city, state, zip -- 
    # but we're using the coordinates endpoint instead, one-at-a-time since batch
    # geocoder only works with addresses. Use individual coordinate lookups instead.
    batch_results = []
    for _, row in batch.iterrows():
        try:
            params = {
                'x': row['longitude'],
                'y': row['latitude'],
                'benchmark': 'Public_AR_Current',
                'vintage': 'Current_Current',
                'format': 'json'
            }
            resp = requests.get(GEOCODER_URL, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    geo = data['result']['geographies']['Census Tracts'][0]
                    geoid_tract = geo['GEOID']  # 11-digit
                    bg = data['result']['geographies'].get('Census Block Groups', [{}])[0]
                    geoid_bg = bg.get('GEOID', geoid_tract + '0')  # 12-digit
                    batch_results.append({
                        'case_id': row['case_id'],
                        'geoid_tract': geoid_tract,
                        'geoid_bg': geoid_bg,
                        'source': 'geocoder'
                    })
                except (KeyError, IndexError):
                    batch_results.append({'case_id': row['case_id'], 'geoid_tract': None, 'geoid_bg': None, 'source': 'no_match'})
            else:
                batch_results.append({'case_id': row['case_id'], 'geoid_tract': None, 'geoid_bg': None, 'source': f'http_{resp.status_code}'})
        except Exception as e:
            batch_results.append({'case_id': row['case_id'], 'geoid_tract': None, 'geoid_bg': None, 'source': 'error'})
        time.sleep(0.05)  # gentle rate limit

    results.extend(batch_results)
    matched = sum(1 for r in batch_results if r['geoid_tract'])
    print(f"    Matched: {matched}/{len(batch_results)}")

    # Save incrementally in case of interruption
    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

final = pd.DataFrame(results)
matched_total = final['geoid_tract'].notna().sum()
print(f"\n[*] Done. {matched_total}/{len(final)} cases geocoded successfully.")
print(f"[*] Written to: {OUT_PATH}")
