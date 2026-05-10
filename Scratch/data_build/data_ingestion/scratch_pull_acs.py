"""
Phase 1: Pull all Travis County ACS tract-level data for vintages 2009-2023.
No API key needed. Public Census API endpoint.
Output: Data/Panel/acs_tract_timeseries.csv
"""
import requests, csv, time, os, json

ROOT = r'C:\Users\dhl\data\thesis\thesis'
OUT_PATH = os.path.join(ROOT, 'Data', 'Panel', 'acs_tract_timeseries.csv')

CENSUS_API_BASE = "https://api.census.gov/data"
STATE = "48"    # Texas
COUNTY = "453"  # Travis County

ACS_VARIABLES = {
    "B01003_001E": "total_population",
    "B01002_001E": "median_age",
    "B02001_002E": "race_white",
    "B02001_003E": "race_black",
    "B02001_005E": "race_asian",
    "B03003_003E": "race_hispanic",
    "B19013_001E": "median_household_income",
    "B17001_002E": "poverty_count",
    "B25077_001E": "median_home_value",
    "B25003_002E": "owner_occupied_units",
    "B25003_003E": "renter_occupied_units",
    "B25064_001E": "median_gross_rent",
    "B25001_001E": "total_housing_units",
}

VINTAGES = list(range(2009, 2024))
var_list = ",".join(ACS_VARIABLES.keys())
friendly = list(ACS_VARIABLES.values())
BAD_VALS = {'-666666666', '-888888888', '-999999999', 'null', 'None', '', None}

all_records = []

print(f"[*] Pulling ACS 5-year estimates for Travis County (FIPS 48453), vintages 2009-2023...")

for vintage in VINTAGES:
    url = (
        f"{CENSUS_API_BASE}/{vintage}/acs/acs5"
        f"?get=NAME,{var_list}"
        f"&for=tract:*"
        f"&in=state:{STATE}&in=county:{COUNTY}"
    )
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            headers = data[0]
            rows = data[1:]
            state_idx = headers.index("state")
            county_idx = headers.index("county")
            tract_idx = headers.index("tract")
            var_idx = {v: headers.index(v) for v in ACS_VARIABLES if v in headers}

            for row in rows:
                geoid = f"{row[state_idx]}{row[county_idx]}{row[tract_idx]}"
                rec = {'geoid_tract': geoid, 'vintage': vintage}
                for api_var, fname in ACS_VARIABLES.items():
                    val = row[var_idx[api_var]] if api_var in var_idx else None
                    rec[fname] = None if str(val) in BAD_VALS else val
                all_records.append(rec)

            print(f"  {vintage}: {len(rows)} tracts")
        else:
            print(f"  {vintage}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  {vintage}: ERROR {e}")

    time.sleep(0.3)

# Write output
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fieldnames = ['geoid_tract', 'vintage'] + friendly
with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_records)

print(f"\n[*] Done. {len(all_records)} tract-vintage records written to {OUT_PATH}")
