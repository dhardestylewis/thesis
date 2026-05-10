"""
scratch_multiparcel_agg.py
============================
For H0 cases that failed single-TCAD join, build aggregated appraisal stats
from case_buffer_map.csv neighboring parcels and from zoning_cases_prefetched_full TCAD IDs.

Strategy:
  1. Primary: zoning_cases_prefetched_full tcad_id (direct subject parcel)
  2. Secondary: case_buffer_map neighbor parcels -> median-aggregate EARS/LDB data
  3. This handles multi-parcel cases and cases where main parcel TCAD ID differs

Output: Data/Panel/case_parcel_agg.csv (case_number -> aggregated appraisal stats)
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath('tcad_normalize.py')))
from tcad_normalize import normalize_tcad_to_9, normalize_ears_to_9

ROOT     = r'C:\Users\dhl\data\thesis\thesis'
INTERIM  = os.path.join(ROOT, 'Data', 'Panel', 'Intermediate')
COA_DIR  = os.path.join(ROOT, 'Data', 'CoA_Open_Data')
ZC_DIR   = os.path.join(ROOT, 'Data', 'Zoning_Cases')

# ── Load case->TCAD mappings ───────────────────────────────────────────────────
print("[1/5] Loading case-parcel linkage files...")

# Primary: zoning_cases_prefetched_full (direct subject parcel TCAD IDs)
zc = pd.read_csv(os.path.join(ZC_DIR, 'Source_Data', 'zoning_cases_prefetched_full.csv'),
                 low_memory=False, usecols=['case_number', 'tcad_id'])
zc = zc[zc['tcad_id'].notna()].copy()
zc['tcad_id'] = zc['tcad_id'].apply(normalize_tcad_to_9)
zc['source'] = 'subject'
print(f"    zoning_cases subject parcels: {len(zc)} rows, {zc.case_number.nunique()} cases")

# Secondary: case_buffer_map (neighboring parcels)
cbm = pd.read_csv(os.path.join(ROOT, 'Data', 'Warehouse_As_Of', 'Build', 'case_buffer_map.csv'),
                  low_memory=False)
cbm['neighbor_tcad_id'] = cbm['neighbor_tcad_id'].apply(normalize_tcad_to_9)
cbm.columns = ['case_number', 'tcad_id']
cbm['source'] = 'buffer'
print(f"    case_buffer_map neighbor parcels: {len(cbm)} rows, {cbm.case_number.nunique()} cases")

# Multi-parcel file: additional subject parcels for specific cases
mp = pd.read_csv(os.path.join(ZC_DIR, 'Processed_Data', 'CSV', 'multi_parcel_closed_2018_2025.csv'),
                 low_memory=False, usecols=['CASE_NUMBER', 'TCAD_ID'])
mp = mp[mp['TCAD_ID'].notna()].copy()
mp.columns = ['case_number', 'tcad_id']
mp['tcad_id'] = mp['tcad_id'].apply(normalize_tcad_to_9)
mp['source'] = 'multi_parcel'
print(f"    multi_parcel file: {len(mp)} rows, {mp.case_number.nunique()} cases")

# Combine all
all_links = pd.concat([zc, cbm, mp], ignore_index=True).drop_duplicates(subset=['case_number', 'tcad_id'])
print(f"    Combined: {len(all_links)} case-parcel links, {all_links.case_number.nunique()} unique cases")

# ── Build EARS/LDB lookup ──────────────────────────────────────────────────────
print("[2/5] Building appraisal lookup (all years)...")

AGG_COLS = ['appraised_value', 'total_market_value', 'land_market_value',
            'improvement_market_value', 'year_built', 'land_acres', 'improvement_sq_ft']

# LDB 2016 and 2021
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
    tmp = tmp.set_index('_join_key')
    appraisal_by_year[yr] = tmp
    print(f"    LDB_{yr}: {len(tmp):,} parcels")

# EARS annual files
EARS_USE = ['account_number', 'appraised_value', 'total_market_value', 'land_market_value',
            'improvement_market_value', 'year_built', 'land_acres', 'improvement_sq_ft']
for fname in sorted(os.listdir(INTERIM)):
    if not fname.startswith('ears_') or not fname.endswith('_clean.csv'): continue
    yr = int(fname.replace('ears_', '').replace('_clean.csv', ''))
    cols_avail = pd.read_csv(os.path.join(INTERIM, fname), nrows=0).columns.tolist()
    use = [c for c in EARS_USE if c in cols_avail]
    tmp = pd.read_csv(os.path.join(INTERIM, fname), usecols=use, low_memory=False)
    tmp['_join_key'] = tmp['account_number'].apply(normalize_ears_to_9)
    tmp = tmp.set_index('_join_key')
    appraisal_by_year[yr] = tmp
    print(f"    ears_{yr}: {len(tmp):,} parcels")

sorted_years = sorted(appraisal_by_year.keys())

def get_appraisal_for_parcel(tcad_id, case_year):
    """Get best appraisal record for a single parcel."""
    # Pre-2019: use LDB 2016 (or earliest)
    if case_year < 2019:
        for y in sorted_years:
            if y <= case_year:
                best_ldb = y
        best_ldb = sorted_years[0] if case_year < sorted_years[0] else best_ldb
        if best_ldb in appraisal_by_year and tcad_id in appraisal_by_year[best_ldb].index:
            rec = appraisal_by_year[best_ldb].loc[tcad_id]
            return rec.iloc[0] if isinstance(rec, pd.DataFrame) else rec
    # 2019+: EARS forward-fill
    best = None
    for y in sorted_years:
        if y <= case_year:
            best = y
    if best is None:
        best = sorted_years[0]
    if tcad_id in appraisal_by_year[best].index:
        rec = appraisal_by_year[best].loc[tcad_id]
        return rec.iloc[0] if isinstance(rec, pd.DataFrame) else rec
    # Fallback: any year
    for y in reversed(sorted_years):
        if tcad_id in appraisal_by_year[y].index:
            rec = appraisal_by_year[y].loc[tcad_id]
            return rec.iloc[0] if isinstance(rec, pd.DataFrame) else rec
    return None

# ── Aggregate per case ─────────────────────────────────────────────────────────
print("[3/5] Aggregating appraisal data per case...")

# Load H0 year lookup
df = pd.read_csv(os.path.join(ROOT, 'Data', 'Warehouse_As_Of', 'H0_Filing_Master_Enriched.csv'),
                 low_memory=False, usecols=['case_number', 'year'])
df = df[df['year'].notna()].copy()
df['year'] = df['year'].astype(int)
case_year_map = dict(zip(df['case_number'].astype(str), df['year'].astype(int)))

case_results = []
cases_processed = 0
cases_matched = 0

for case_number, group in all_links.groupby('case_number'):
    case_year = case_year_map.get(str(case_number))
    if not case_year:
        continue

    parcel_records = []
    for _, row in group.iterrows():
        tcad_id = normalize_tcad_to_9(str(row['tcad_id']))
        if not tcad_id:
            continue
        rec = get_appraisal_for_parcel(tcad_id, case_year)
        if rec is not None:
            rec_dict = rec.to_dict() if hasattr(rec, 'to_dict') else dict(rec)
            rec_dict['_source'] = row['source']
            parcel_records.append(rec_dict)

    if parcel_records:
        pr_df = pd.DataFrame(parcel_records)
        # Aggregate: median for financial/size, mode for year_built
        agg = {'case_number': str(case_number), 'n_parcels': len(parcel_records)}
        for col in AGG_COLS:
            if col in pr_df.columns:
                nums = pd.to_numeric(pr_df[col], errors='coerce').dropna()
                if len(nums) > 0:
                    agg[f'agg_{col}_median'] = nums.median()
                    agg[f'agg_{col}_total'] = nums.sum() if 'value' in col or 'acres' in col or 'sqft' in col else None
        agg['agg_source_priority'] = 'subject' if any(r['_source'] == 'subject' for r in parcel_records) else 'buffer'
        case_results.append(agg)
        cases_matched += 1

    cases_processed += 1
    if cases_processed % 500 == 0:
        print(f"    {cases_processed} cases processed, {cases_matched} matched")

print(f"\n    Done: {cases_matched}/{cases_processed} cases with appraisal data via multi-parcel aggregation")

# ── Save ───────────────────────────────────────────────────────────────────────
print("[4/5] Saving...")
agg_df = pd.DataFrame(case_results)
out_path = os.path.join(ROOT, 'Data', 'Panel', 'case_parcel_agg.csv')
agg_df.to_csv(out_path, index=False)
print(f"    {out_path}: {len(agg_df)} cases, {len(agg_df.columns)} cols")

# ── Merge back onto H0 ────────────────────────────────────────────────────────
print("[5/5] Checking coverage improvement against H0...")
h0 = pd.read_csv(os.path.join(ROOT, 'Data', 'Warehouse_As_Of', 'H0_Filing_Master_Enriched.csv'),
                 low_memory=False)
merged = h0.merge(agg_df, left_on='case_number', right_on='case_number', how='left')
print(f"    agg_appraised_value_median coverage: {merged['agg_appraised_value_median'].notna().mean()*100:.1f}%")
print(f"    agg_year_built_median coverage: {merged['agg_year_built_median'].notna().mean()*100:.1f}%")
print(f"    agg_land_acres_total coverage: {merged['agg_land_acres_total'].notna().mean()*100:.1f}%")
print("Done.")
