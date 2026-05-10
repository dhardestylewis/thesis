"""
scratch_spatial_parcel_join.py
================================
For H0 cases missing EARS appraisal data, find nearest parcel centroid via KDTree
and join EARS/LDB using that parcel's ID.

parcel_centroids.csv: 284,791 Austin parcels with lat/lon + parcel_id_10
id_crosswalk.csv: parcel_id_10 -> ears_account_number

Workflow:
  1. Load parcel_centroids + id_crosswalk
  2. Build KDTree on parcel lat/lon
  3. For each H0 case with lat/lon but missing EARS: find nearest parcel
  4. Look up that parcel's appraisal data from EARS/LDB
  5. Append to case_parcel_agg.csv as spatial fallback

Output: Data/Panel/case_parcel_agg_v2.csv (all cases including spatial fallback)
"""
import pandas as pd
import numpy as np
import os, sys
from scipy.spatial import cKDTree

sys.path.insert(0, r'C:\Users\dhl\data\thesis\thesis')
from tcad_normalize import normalize_tcad_to_9, normalize_ears_to_9

ROOT     = r'C:\Users\dhl\data\thesis\thesis'
INTERIM  = os.path.join(ROOT, 'Data', 'Panel', 'Intermediate')
COA_DIR  = os.path.join(ROOT, 'Data', 'CoA_Open_Data')
REF_DIR  = os.path.join(ROOT, 'Data', 'Panel', 'Reference')
WH_DIR   = os.path.join(ROOT, 'Data', 'Warehouse_As_Of')

# ── Load parcel centroids and crosswalk ───────────────────────────────────────
print("[1/6] Loading parcel centroids + crosswalk...")
cents = pd.read_csv(os.path.join(REF_DIR, 'parcel_centroids.csv'), low_memory=False)
cents = cents.dropna(subset=['latitude', 'longitude']).copy()
cents['_pid_norm'] = cents['parcel_id_10'].astype(str).apply(normalize_tcad_to_9)
print(f"    Parcel centroids: {len(cents):,} parcels with lat/lon")

xw = pd.read_csv(os.path.join(REF_DIR, 'id_crosswalk.csv'), low_memory=False)
xw['_pid_norm'] = xw['parcel_id_10'].astype(str).apply(normalize_tcad_to_9)
# crosswalk gives us parcel_id_10 -> ears_account_number (internal EARS ID, not PID_10)
# but parcel_id_10 itself IS the join key for LDB/EARS after normalization
print(f"    Crosswalk: {len(xw):,} parcel->EARS mappings")

# ── Build EARS/LDB single-parcel lookup ───────────────────────────────────────
print("[2/6] Building appraisal lookup...")

EARS_USE = ['account_number', 'appraised_value', 'total_market_value', 'land_market_value',
            'improvement_market_value', 'year_built', 'land_acres', 'improvement_sq_ft']
LDB_COL_MAP_2016 = {
    'PID_10': '_join_key', 'APPRAISED_VAL': 'appraised_value', 'MARKET_VAL': 'total_market_value',
    'YR_BUILT': 'year_built', 'LAND_ACRES': 'land_acres', 'SUM_IMPRV_SQFT': 'improvement_sq_ft',
}
LDB_COL_MAP_2021 = {
    'PID_10': '_join_key', 'APPRAISED_': 'appraised_value', 'MARKET_VAL': 'total_market_value',
    'YR_BUILT': 'year_built', 'LAND_ACRES': 'land_acres', 'SUM_IMPRV_': 'improvement_sq_ft',
}

appraisal_by_year = {}
for yr, fname, col_map in [(2016, 'LDB_2016_4nsn-uea6.csv', LDB_COL_MAP_2016),
                            (2021, 'LDB_2021_kk8y-6cmt.csv', LDB_COL_MAP_2021)]:
    path = os.path.join(COA_DIR, fname)
    if not os.path.exists(path): continue
    cols_avail = pd.read_csv(path, nrows=0, encoding='latin-1').columns.tolist()
    use_cols = [c for c in col_map if c in cols_avail]
    tmp = pd.read_csv(path, usecols=use_cols, low_memory=False, encoding='latin-1')
    tmp = tmp.rename(columns={c: col_map[c] for c in use_cols})
    tmp['_join_key'] = tmp['_join_key'].apply(normalize_tcad_to_9)
    appraisal_by_year[yr] = tmp.set_index('_join_key')
    print(f"    LDB_{yr}: {len(tmp):,} parcels")

for fname in sorted(os.listdir(INTERIM)):
    if not fname.startswith('ears_') or not fname.endswith('_clean.csv'): continue
    yr = int(fname.replace('ears_', '').replace('_clean.csv', ''))
    cols_avail = pd.read_csv(os.path.join(INTERIM, fname), nrows=0).columns.tolist()
    use = [c for c in EARS_USE if c in cols_avail]
    tmp = pd.read_csv(os.path.join(INTERIM, fname), usecols=use, low_memory=False)
    tmp['_join_key'] = tmp['account_number'].apply(normalize_ears_to_9)
    appraisal_by_year[yr] = tmp.set_index('_join_key')
    print(f"    ears_{yr}: {len(tmp):,} parcels")

sorted_years = sorted(appraisal_by_year.keys())

def lookup_parcel(pid_norm, case_year):
    if not pid_norm: return {}
    src = appraisal_by_year[2016 if case_year < 2019 else
          max(y for y in sorted_years if y <= case_year) if any(y <= case_year for y in sorted_years)
          else sorted_years[0]]
    if pid_norm in src.index:
        rec = src.loc[pid_norm]
        return (rec.iloc[0] if isinstance(rec, pd.DataFrame) else rec).to_dict()
    # fallback any year
    for y in reversed(sorted_years):
        if pid_norm in appraisal_by_year[y].index:
            rec = appraisal_by_year[y].loc[pid_norm]
            return (rec.iloc[0] if isinstance(rec, pd.DataFrame) else rec).to_dict()
    return {}

# ── Build KDTree on parcel centroids ─────────────────────────────────────────
print("[3/6] Building KDTree on parcel centroids...")
coords_rad = np.radians(cents[['latitude','longitude']].values)
tree = cKDTree(coords_rad)
print(f"    KDTree built on {len(cents):,} parcel centroids")

EARTH_R_M = 6_371_000  # meters
MAX_DIST_M = 300        # only accept matches within 300m

def find_nearest_parcel(lat, lon):
    pt = np.radians([[lat, lon]])
    dist_rad, idx = tree.query(pt, k=1)
    dist_m = dist_rad[0] * EARTH_R_M
    if dist_m > MAX_DIST_M:
        return None, dist_m
    pid = cents.iloc[idx[0]]['_pid_norm']
    return pid, dist_m

# ── Load H0 and existing agg to find still-missing cases ─────────────────────
print("[4/6] Finding cases still missing appraisal data...")
h0 = pd.read_csv(os.path.join(WH_DIR, 'H0_Filing_Master_Enriched.csv'), low_memory=False)
h0 = h0[h0['year'].notna()].copy()
h0['year'] = h0['year'].astype(int)

# Load current agg to avoid re-doing already matched cases
existing_agg_path = os.path.join(ROOT, 'Data', 'Panel', 'case_parcel_agg.csv')
existing_matched = set()
if os.path.exists(existing_agg_path):
    ex = pd.read_csv(existing_agg_path)
    existing_matched = set(ex[ex['agg_appraised_value_median'].notna()]['case_number'].astype(str))
    print(f"    Already matched: {len(existing_matched)} cases from prior agg")

# Cases with lat/lon but still missing
has_latlon = h0['latitude'].notna() & h0['longitude'].notna()
still_missing = h0[has_latlon & ~h0['case_number'].astype(str).isin(existing_matched)].copy()
print(f"    Cases with lat/lon but missing appraisal: {len(still_missing)}")

# ── Spatial join ─────────────────────────────────────────────────────────────
print("[5/6] Spatial nearest-parcel join...")
spatial_results = []
matched = 0
skipped_dist = 0

for _, row in still_missing.iterrows():
    lat, lon = row['latitude'], row['longitude']
    case_year = int(row['year'])
    case_num = str(row['case_number'])

    pid, dist_m = find_nearest_parcel(lat, lon)
    if pid is None:
        skipped_dist += 1
        continue

    rec = lookup_parcel(pid, case_year)
    if rec:
        agg_rec = {
            'case_number': case_num,
            'n_parcels': 1,
            'agg_appraised_value_median': rec.get('appraised_value'),
            'agg_total_market_value_median': rec.get('total_market_value'),
            'agg_land_market_value_median': rec.get('land_market_value'),
            'agg_improvement_market_value_median': rec.get('improvement_market_value'),
            'agg_year_built_median': rec.get('year_built'),
            'agg_land_acres_median': rec.get('land_acres'),
            'agg_land_acres_total': rec.get('land_acres'),
            'agg_improvement_sq_ft_median': rec.get('improvement_sq_ft'),
            'agg_improvement_sq_ft_total': None,
            'agg_source_priority': f'spatial_{dist_m:.0f}m',
        }
        spatial_results.append(agg_rec)
        matched += 1

print(f"    Spatial matched: {matched} cases")
print(f"    Beyond {MAX_DIST_M}m threshold: {skipped_dist} cases")

# ── Save merged agg ───────────────────────────────────────────────────────────
print("[6/6] Saving combined agg (prior + spatial)...")
spatial_df = pd.DataFrame(spatial_results)
if os.path.exists(existing_agg_path):
    prior = pd.read_csv(existing_agg_path)
    combined = pd.concat([prior, spatial_df], ignore_index=True)
else:
    combined = spatial_df

out_path = os.path.join(ROOT, 'Data', 'Panel', 'case_parcel_agg_v2.csv')
combined.to_csv(out_path, index=False)
print(f"    {out_path}: {len(combined)} cases")

# Coverage check
merged = h0.merge(combined, on='case_number', how='left')
cov = merged['agg_appraised_value_median'].notna().mean() * 100
print(f"    Final appraisal coverage across H0: {cov:.1f}%")
print("Done.")
