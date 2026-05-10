"""
scratch_rebuild_enriched.py
============================
Rebuilds H0_Filing_Master_Enriched_v2.csv using ALL available sources:
  - EARS annual appraisal files (2019-2025) for parcel-level TCAD data
  - ACS tract timeseries (just pulled from Census API)
  - GEOID lookup via case_geoid_lookup.csv (from geocoder) OR via parcel_centroids join
  - LDB 2016 + 2021 for land use / zoning code forward-fill
  - Original H0 features (procedural delta features, lat/lon, etc.)

Output: Data/Warehouse_As_Of/H0_Filing_Master_Enriched_v2.csv
"""
import pandas as pd
import numpy as np
import os

ROOT = r'C:\Users\dhl\data\thesis\thesis'
PANEL_DIR = os.path.join(ROOT, 'Data', 'Panel')
INTERIM_DIR = os.path.join(PANEL_DIR, 'Intermediate')
COA_DIR = os.path.join(ROOT, 'Data', 'CoA_Open_Data')
REF_DIR = os.path.join(PANEL_DIR, 'Reference')
WH_DIR = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

print("[1/7] Loading base H0 feature set...")
sc = pd.read_parquet(os.path.join(ROOT, 'Data', 'interim', 'stage_c_features_raw.parquet'))
sc['year'] = pd.to_datetime(sc['as_of_date']).dt.year
print(f"    {len(sc)} cases, years {sc['year'].min()}-{sc['year'].max()}")

# ─── Step 1: GEOID lookup ─────────────────────────────────────────────────────
print("[2/7] Building GEOID lookup...")

# Source A: geocoder output (if it exists/ran partially)
geoid_lookup = {}
geoid_path = os.path.join(PANEL_DIR, 'case_geoid_lookup.csv')
if os.path.exists(geoid_path):
    gl = pd.read_csv(geoid_path)
    for _, row in gl.iterrows():
        if pd.notna(row.get('geoid_tract')):
            geoid_lookup[str(row['case_id'])] = row['geoid_tract']
    print(f"    Geocoder: {len(geoid_lookup)} GEOIDs from case-level geocoder")

# Source B: parcel_centroids + ACS using lat/lon → nearest parcel → TCAD ID → GEOID
# We use the already-populated GEOIDs in stage_c where available
geoid_from_sc = 0
for _, row in sc.iterrows():
    cid = str(row['case_id'])
    if cid not in geoid_lookup:
        g = row.get('nearby_GEOID') or row.get('zoning_case_GEOID')
        if pd.notna(g) and str(g).strip():
            geoid_lookup[cid] = str(g)[:11]  # truncate to tract
            geoid_from_sc += 1
print(f"    Stage_c existing GEOIDs: {geoid_from_sc} additional")
print(f"    Total GEOID coverage: {len(geoid_lookup)} / {len(sc)} ({len(geoid_lookup)/len(sc)*100:.1f}%)")

# ─── Step 2: ACS data join ────────────────────────────────────────────────────
print("[3/7] Loading ACS tract timeseries...")
acs_path = os.path.join(PANEL_DIR, 'acs_tract_timeseries.csv')
acs = pd.read_csv(acs_path)
acs['vintage'] = acs['vintage'].astype(int)
acs['geoid_tract'] = acs['geoid_tract'].astype(str)
print(f"    {len(acs)} tract-vintage records, vintages {acs['vintage'].min()}-{acs['vintage'].max()}")

# Build lookup: (geoid_tract, vintage) -> record
acs_cols = [c for c in acs.columns if c not in ('geoid_tract', 'vintage')]
acs_dict = {}
for _, row in acs.iterrows():
    acs_dict[(row['geoid_tract'], int(row['vintage']))] = row[acs_cols].to_dict()

def get_acs(case_id, case_year):
    geoid = geoid_lookup.get(str(case_id))
    if not geoid:
        return {}
    geoid = str(geoid)[:11]
    # Find best vintage <= case_year
    best = None
    for v in range(int(case_year), 2008, -1):
        key = (geoid, v)
        if key in acs_dict:
            best = acs_dict[key]
            break
    return best or {}

# ─── Step 3: EARS appraisal join ─────────────────────────────────────────────
print("[4/7] Loading EARS annual appraisal files...")
ears_by_year = {}
EARS_COLS = ['account_number', '_parcel_id_10', 'appraised_value', 'assessed_value',
             'total_market_value', 'land_market_value', 'improvement_market_value',
             'year_built', 'land_acres', 'deed_acreage', 'improvement_sq_ft',
             'land_use_code', 'zoning_code', 'ears_year']
for fname in sorted(os.listdir(INTERIM_DIR)):
    if not fname.startswith('ears_') or not fname.endswith('_clean.csv'):
        continue
    yr = int(fname.replace('ears_', '').replace('_clean.csv', ''))
    cols_avail = pd.read_csv(os.path.join(INTERIM_DIR, fname), nrows=0).columns.tolist()
    use_cols = [c for c in EARS_COLS if c in cols_avail]
    df = pd.read_csv(os.path.join(INTERIM_DIR, fname), usecols=use_cols, low_memory=False)
    # Index by both account_number AND _parcel_id_10
    df['_parcel_id_10'] = df['_parcel_id_10'].fillna(0).astype(float).astype(int).astype(str)
    df['account_number'] = df['account_number'].astype(str)
    ears_by_year[yr] = df
    print(f"    {fname}: {len(df):,} parcels")

ears_years = sorted(ears_by_year.keys())

def get_ears(tcad_id, pid10, case_year):
    """Forward-fill: find nearest EARS year <= case_year."""
    tcad_id = str(tcad_id) if pd.notna(tcad_id) else None
    pid10 = str(int(float(pid10))) if pd.notna(pid10) and str(pid10).strip() else None
    best_year = None
    for y in ears_years:
        if y <= case_year:
            best_year = y
    if best_year is None:
        best_year = ears_years[0]
    df = ears_by_year[best_year]
    # Try account_number first, then pid_10
    if tcad_id:
        row = df[df['account_number'] == tcad_id]
        if len(row) > 0:
            return row.iloc[0].to_dict()
    if pid10:
        row = df[df['_parcel_id_10'] == pid10]
        if len(row) > 0:
            return row.iloc[0].to_dict()
    return {}

# ─── Step 4: Build enriched dataset ──────────────────────────────────────────
print("[5/7] Joining everything onto H0 cases...")

acs_matched = 0
ears_matched = 0
records = []

sc = sc[sc['year'].notna() & sc['year'].between(2007, 2025)].copy()
sc['year'] = sc['year'].astype(int)

for i, row in sc.iterrows():
    case_id = str(row['case_id'])
    case_year = int(row['year'])
    tcad_id = str(row.get('account_number_formatted', '')) if pd.notna(row.get('account_number_formatted')) else None
    pid10 = str(row.get('_parcel_id_10', '')) if '_parcel_id_10' in row.index and pd.notna(row.get('_parcel_id_10')) else None

    rec = row.to_dict()

    # ACS join
    acs_rec = get_acs(case_id, case_year)
    if acs_rec:
        for k, v in acs_rec.items():
            rec[f'acs_{k}'] = v
        rec['acs_geoid'] = geoid_lookup.get(case_id)
        rec['acs_vintage_used'] = case_year
        acs_matched += 1
    else:
        rec['acs_geoid'] = None
        rec['acs_vintage_used'] = None

    # EARS join
    ears_rec = get_ears(tcad_id, pid10, case_year)
    if ears_rec:
        for k, v in ears_rec.items():
            if k not in ('account_number', '_parcel_id_10', 'ears_year'):
                rec[f'ears_{k}'] = v
        ears_matched += 1

    records.append(rec)

    if (i + 1) % 500 == 0:
        print(f"    Processed {i+1}/{len(sc)} cases... (ACS: {acs_matched}, EARS: {ears_matched})")

print(f"\n    Final: ACS matched {acs_matched}/{len(sc)} ({acs_matched/len(sc)*100:.1f}%)")
print(f"    Final: EARS matched {ears_matched}/{len(sc)} ({ears_matched/len(sc)*100:.1f}%)")

# ─── Step 5: Save ────────────────────────────────────────────────────────────
print("[6/7] Saving enriched dataset...")
out = pd.DataFrame(records)
out_path = os.path.join(WH_DIR, 'H0_Filing_Master_Enriched_v2.csv')
out.to_csv(out_path, index=False)
sz = os.path.getsize(out_path) / 1e6
print(f"    Written: {out_path} ({sz:.1f} MB, {len(out)} rows, {len(out.columns)} columns)")

# ─── Step 6: Null rate report ─────────────────────────────────────────────────
print("[7/7] Null rate report for key features...")
key_features = [
    'acs_median_household_income', 'acs_renter_occupied_units', 'acs_owner_occupied_units',
    'acs_total_population', 'acs_median_home_value',
    'ears_appraised_value', 'ears_year_built', 'ears_land_acres',
    'ears_improvement_sq_ft', 'ears_total_market_value',
    'gross_site_area_acres', 'latitude', 'longitude'
]
for f in key_features:
    if f in out.columns:
        pct = out[f].isna().mean() * 100
        print(f"    {f}: {pct:.1f}% null")
    else:
        print(f"    {f}: NOT IN OUTPUT")

print("\nDone.")
