import pandas as pd, requests, os, time

GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
OUT_PATH = 'Data/Panel/geo/case_geoid_lookup.csv'

# Load base 2007-2025 cases
sc = pd.read_parquet('Data/interim/stage_c_features_raw.parquet')
sc['year'] = pd.to_datetime(sc['as_of_date']).dt.year
cases = sc[sc['year'].notna() & sc['year'].between(2007, 2025)].copy()
has_ll = cases['latitude'].notna() & cases['longitude'].notna()
cases = cases[has_ll][['case_id', 'latitude', 'longitude']].reset_index(drop=True)

# Load already geocoded
done_ids = set()
if os.path.exists(OUT_PATH):
    done = pd.read_csv(OUT_PATH)
    done_ids = set(done['case_id'].astype(str))

# Filter to missing
missing = cases[~cases['case_id'].astype(str).isin(done_ids)].copy()
print(f"[*] {len(done_ids)} cases already geocoded.")
print(f"[*] Geocoding {len(missing)} remaining target cases...")

results = []
for i, row in missing.iterrows():
    try:
        params = {'x': row['longitude'], 'y': row['latitude'], 'benchmark': 'Public_AR_Current', 'vintage': 'Current_Current', 'format': 'json'}
        resp = requests.get(GEOCODER_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            try:
                geoid_tract = data['result']['geographies']['Census Tracts'][0]['GEOID']
                geoid_bg = data['result']['geographies'].get('Census Block Groups', [{}])[0].get('GEOID', geoid_tract+'0')
                results.append({'case_id': row['case_id'], 'geoid_tract': geoid_tract, 'geoid_bg': geoid_bg, 'source': 'geocoder'})
            except:
                pass
    except:
        pass
    
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(missing)} processed...")
        # Save incremental
        pd.DataFrame(results).to_csv(OUT_PATH, mode='a', header=False, index=False)
        results = []

if results:
    pd.DataFrame(results).to_csv(OUT_PATH, mode='a', header=False, index=False)
print("[*] Done.")
