"""
scratch_rebuild_enriched_v2.py
================================
Correct rebuild using H0_Filing_Master_Enriched.csv as base (has standardized_tcad_id at 83%)
Joins EARS via standardized_tcad_id matching EARS account_number (with format normalization).
Joins ACS via nearby_GEOID/zoning_case_GEOID.
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath('tcad_normalize.py')))
from tcad_normalize import normalize_tcad_to_9, normalize_ears_to_9

ROOT     = r'C:\Users\dhl\data\thesis\thesis'
WH_DIR   = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')
PANEL_DIR = os.path.join(ROOT, 'Data', 'Panel')
INTERIM  = os.path.join(PANEL_DIR, 'Intermediate')
REF_DIR  = os.path.join(PANEL_DIR, 'Reference')

# ─── Base dataset ─────────────────────────────────────────────────────────────
print("[1/6] Loading H0_Filing_Master_Enriched.csv...")
df = pd.read_csv(os.path.join(WH_DIR, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df[df['year'].notna() & df['year'].between(2007, 2025)].copy()
df['year'] = df['year'].astype(int)
print(f"    {len(df)} cases, years {df['year'].min()}-{df['year'].max()}")
print(f"    Columns: {len(df.columns)}")

# Normalize TCAD join key to 9-digit canonical form
df['_join_tcad'] = df['standardized_tcad_id'].apply(normalize_tcad_to_9)
tcad_coverage = df['_join_tcad'].notna().sum()
print(f"    TCAD IDs for join: {tcad_coverage} ({tcad_coverage/len(df)*100:.1f}%)")

# ─── EARS annual appraisal join ───────────────────────────────────────────────
print("[2/6] Building EARS lookup by year (account_number indexed)...")

EARS_USE_COLS = [
    'account_number', '_parcel_id_10', 'ears_year',
    'appraised_value', 'assessed_value', 'total_market_value',
    'land_market_value', 'improvement_market_value',
    'year_built', 'land_acres', 'deed_acreage', 'improvement_sq_ft',
    'land_use_code', 'zoning_code'
]

# Load LDB_2016 as appraisal source for pre-2019 cases (same PID_10 join key)
print("[2a/6] Loading LDB_2016 as pre-2019 appraisal source...")
LDB_COL_MAP_2016 = {
    'PID_10': '_join_key', 'APPRAISED_VAL': 'appraised_value', 'MARKET_VAL': 'total_market_value',
    'YR_BUILT': 'year_built', 'LAND_ACRES': 'land_acres', 'SUM_IMPRV_SQFT': 'improvement_sq_ft',
    'FAR': 'ldb_far', 'ILR': 'ldb_ilr', 'UNITS': 'ldb_units', 'BASEZONE': 'ldb_basezone',
    'CONSTRAINED_AREA': 'ldb_constrained_area'
}
LDB_COL_MAP_2021 = {
    'PID_10': '_join_key', 'APPRAISED_': 'appraised_value', 'MARKET_VAL': 'total_market_value',
    'YR_BUILT': 'year_built', 'LAND_ACRES': 'land_acres', 'SUM_IMPRV_': 'improvement_sq_ft',
    'FAR': 'ldb_far', 'ILR': 'ldb_ilr', 'UNITS': 'ldb_units', 'BASEZONE': 'ldb_basezone',
    'CONSTRAINE': 'ldb_constrained_area'
}
ldb_by_year = {}
for ldb_year, ldb_fname, col_map in [
    (2016, 'LDB_2016_4nsn-uea6.csv', LDB_COL_MAP_2016),
    (2021, 'LDB_2021_kk8y-6cmt.csv', LDB_COL_MAP_2021)
]:
    ldb_path = os.path.join(ROOT, 'Data', 'CoA_Open_Data', ldb_fname)
    if not os.path.exists(ldb_path):
        print(f"    LDB_{ldb_year}: NOT FOUND, skipping")
        continue
    cols_avail = pd.read_csv(ldb_path, nrows=0, encoding='latin-1').columns.tolist()
    use_cols = [c for c in col_map if c in cols_avail]
    ldb_raw = pd.read_csv(ldb_path, usecols=use_cols, low_memory=False, encoding='latin-1')
    ldb_raw = ldb_raw.rename(columns={c: col_map[c] for c in use_cols})
    ldb_raw['_join_key'] = ldb_raw['_join_key'].apply(normalize_tcad_to_9)
    ldb_by_year[ldb_year] = ldb_raw.set_index('_join_key')
    print(f"    LDB_{ldb_year}: {len(ldb_raw):,} parcels")
ldb_years_sorted = sorted(ldb_by_year.keys())

ears_by_year = {}
for fname in sorted(os.listdir(INTERIM)):
    if not fname.startswith('ears_') or not fname.endswith('_clean.csv'):
        continue
    yr = int(fname.replace('ears_', '').replace('_clean.csv', ''))
    cols_avail = pd.read_csv(os.path.join(INTERIM, fname), nrows=0).columns.tolist()
    use = [c for c in EARS_USE_COLS if c in cols_avail]
    tmp = pd.read_csv(os.path.join(INTERIM, fname), usecols=use, low_memory=False)
    tmp['_join_key'] = tmp['account_number'].apply(normalize_ears_to_9)
    ears_by_year[yr] = tmp.set_index('_join_key')
    print(f"    ears_{yr}: {len(tmp):,} parcels")

ears_years_sorted = sorted(ears_by_year.keys())

def lookup_ears(tcad_id, case_year):
    """For case_year < 2019: use LDB_2016 forward-fill. For 2019+: use nearest EARS year."""
    tcad_id = normalize_tcad_to_9(tcad_id)
    if not tcad_id:
        return {}

    # Pre-2019: use LDB snapshots (2016 = nearest available, also covers <=2016)
    if case_year < 2019:
        best_ldb = None
        for y in ldb_years_sorted:
            if y <= case_year:
                best_ldb = y
        if best_ldb is None:
            best_ldb = ldb_years_sorted[0]  # earliest available (2016)
        lkp = ldb_by_year[best_ldb]
        if tcad_id in lkp.index:
            rec = lkp.loc[tcad_id]
            if isinstance(rec, pd.DataFrame): rec = rec.iloc[0]
            return rec.to_dict()
        # Fall through to EARS if LDB misses

    # 2019+: use nearest EARS year <= case_year
    best = None
    for y in ears_years_sorted:
        if y <= case_year:
            best = y
    if best is None:
        best = ears_years_sorted[0]
    lkp = ears_by_year[best]
    if tcad_id in lkp.index:
        rec = lkp.loc[tcad_id]
        if isinstance(rec, pd.DataFrame): rec = rec.iloc[0]
        return rec.to_dict()

    # Last resort: try LDB for any year
    if case_year >= 2019:
        for ly in reversed(ldb_years_sorted):
            lkp = ldb_by_year[ly]
            if tcad_id in lkp.index:
                rec = lkp.loc[tcad_id]
                if isinstance(rec, pd.DataFrame): rec = rec.iloc[0]
                return rec.to_dict()
    return {}

# ─── ACS join ─────────────────────────────────────────────────────────────────
print("[3/6] Loading ACS timeseries & Geocoder output...")
acs = pd.read_csv(os.path.join(PANEL_DIR, 'acs_tract_timeseries.csv'))
acs['vintage'] = acs['vintage'].astype(int)
acs['geoid_tract'] = acs['geoid_tract'].astype(str)
acs_cols = [c for c in acs.columns if c not in ('geoid_tract', 'vintage')]
acs_dict = {}
for _, row in acs.iterrows():
    acs_dict[(row['geoid_tract'], int(row['vintage']))] = row[acs_cols].to_dict()

geoid_lookup = {}
geoid_path = os.path.join(PANEL_DIR, 'case_geoid_lookup.csv')
if os.path.exists(geoid_path):
    gdf = pd.read_csv(geoid_path)
    for _, row in gdf.iterrows():
        if pd.notna(row.get('geoid_tract')):
            geoid_lookup[str(row['case_id'])] = str(row['geoid_tract'])
print(f"    Loaded {len(geoid_lookup)} GEOIDs from geocoder cache.")

def lookup_acs(geoid, case_year):
    if not geoid or str(geoid) in ('nan', '', 'None'):
        return {}
    g = str(geoid)[:11]
    # Backward search (nearest past)
    for v in range(int(case_year), 2008, -1):
        rec = acs_dict.get((g, v))
        if rec:
            return rec
    # Forward search (nearest future) if backward fails
    for v in range(int(case_year) + 1, 2025):
        rec = acs_dict.get((g, v))
        if rec:
            return rec
    return {}

# ─── Join EARS and ACS onto each case ─────────────────────────────────────────
print("[4/6] Joining EARS and ACS onto all cases...")

ears_matched = 0
acs_matched  = 0
ears_prefix_cols = [c for c in EARS_USE_COLS if c not in ('account_number', '_parcel_id_10', 'ears_year')]
acs_prefix_cols = acs_cols

# Build new columns
new_ears_rows = []
new_acs_rows = []

for i, row in df.iterrows():
    tcad  = row.get('_join_tcad', '')
    year  = int(row['year'])
    case_id_str = str(row['case_id']) if 'case_id' in row else str(row['case_number'])
    geoid = geoid_lookup.get(case_id_str) or row.get('nearby_GEOID') or row.get('zoning_case_GEOID')

    # EARS
    er = lookup_ears(tcad, year)
    ears_rec = {f'ears_{c}': er.get(c, np.nan) for c in ears_prefix_cols}
    if er:
        ears_matched += 1
    new_ears_rows.append(ears_rec)

    # ACS
    ar = lookup_acs(geoid, year)
    acs_rec = {f'acs2_{c}': ar.get(c, np.nan) for c in acs_prefix_cols}
    if ar:
        acs_matched += 1
    new_acs_rows.append(acs_rec)

    if (i + 1) % 1000 == 0 or i == len(df) - 1:
        print(f"    {i+1}/{len(df)}... EARS={ears_matched}, ACS={acs_matched}")

# Append new columns
ears_df = pd.DataFrame(new_ears_rows, index=df.index)
acs_df  = pd.DataFrame(new_acs_rows,  index=df.index)
df = pd.concat([df, ears_df, acs_df], axis=1)
df.drop(columns=['_join_tcad'], inplace=True, errors='ignore')

print(f"\n    EARS matched: {ears_matched}/{len(df)} ({ears_matched/len(df)*100:.1f}%)")
print(f"    ACS matched:  {acs_matched}/{len(df)} ({acs_matched/len(df)*100:.1f}%)")

# ─── Multi-parcel aggregate join ──────────────────────────────────────────────
print("[4b/6] Merging multi-parcel aggregated appraisal stats...")
agg_path = os.path.join(PANEL_DIR, 'case_parcel_agg.csv')
if os.path.exists(agg_path):
    agg = pd.read_csv(agg_path, low_memory=False)
    # Only use agg columns for cases where ears_ is null
    df = df.merge(agg, on='case_number', how='left')
    # For each key appraisal field: fill ears_X from agg_X_median where ears_X is null
    fill_map = {
        'ears_appraised_value':     'agg_appraised_value_median',
        'ears_total_market_value':  'agg_total_market_value_median',
        'ears_land_acres':          'agg_land_acres_median',
        'ears_improvement_sq_ft':   'agg_improvement_sq_ft_median',
        'ears_year_built':          'agg_year_built_median',
    }
    for primary, fallback in fill_map.items():
        if primary in df.columns and fallback in df.columns:
            df[primary] = df[primary].fillna(df[fallback])
    agg_match = df['agg_appraised_value_median'].notna().sum()
    print(f"    Multi-parcel agg coverage: {agg_match}/{len(df)} ({agg_match/len(df)*100:.1f}%)")
else:
    print("    case_parcel_agg.csv not found, skipping")


# ─── Save ──────────────────────────────────────────────────────────────────────
print("[5/6] Saving...")
out_path = os.path.join(WH_DIR, 'H0_Filing_Master_Enriched_v2.csv')
df.to_csv(out_path, index=False)
sz = os.path.getsize(out_path) / 1e6
print(f"    {out_path} ({sz:.1f} MB, {len(df)} rows, {len(df.columns)} cols)")

# ─── Null report ──────────────────────────────────────────────────────────────
print("[6/6] Null rate report:")
key_feats = [
    'ears_appraised_value','ears_year_built','ears_land_acres','ears_improvement_sq_ft',
    'ears_total_market_value',
    'acs2_median_household_income','acs2_renter_occupied_units','acs2_owner_occupied_units',
    'acs2_total_population',
    'gross_site_area_acres','latitude','longitude',
    'ldb_appraised_val','ldb_yr_built','ldb_imprv_sqft'
]
for f in key_feats:
    if f in df.columns:
        pct = df[f].isna().mean() * 100
        print(f"    {f}: {pct:.1f}% null")
    else:
        print(f"    {f}: NOT IN OUTPUT")
print("Done.")
