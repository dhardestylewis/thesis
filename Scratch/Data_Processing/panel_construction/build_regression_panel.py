"""
build_regression_panel.py
Assembles a temporally leakage-free regression panel by joining:
  1. Zoning case trajectories + temporal anchors
  2. Parcel/tax data (LDB 2016, LDB 2021, or EARS by year)
  3. Protest petitions (endogenous, case-level, dated)
  4. Commission/council sentiment (distributed Q-1/Q-2/Q-3 calendar lags)
  5. Census ACS (Lag-1 annual)
"""
import os
import re
import pandas as pd
import numpy as np
from dateutil import parser as dateparser

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = r"c:\Users\dhl\data\Thesis\thesis\Data"
SCRATCH  = r"c:\Users\dhl\data\Thesis\thesis\Scratch"
OUT_DIR  = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

ZONING_CSV     = os.path.join(BASE, "model_ready_zoning_data.csv")
ENRICHED_CSV   = os.path.join(BASE, "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_causal.csv")
PETITION_CSV   = os.path.join(BASE, "Protest_Petitions", "petition_signers_backfilled.csv")
COUNCIL_CSV    = os.path.join(BASE, "council_agendas_cases.csv")
COMMISSION_CSV = os.path.join(BASE, "planning_commission_index.csv")
LDB16_CSV      = os.path.join(BASE, "CoA_Open_Data", "LDB_2016_4nsn-uea6.csv")
LDB21_CSV      = os.path.join(BASE, "CoA_Open_Data", "LDB_2021_kk8y-6cmt.csv")
CROSSWALK_CSV  = os.path.join(BASE, "Panel", "Reference", "id_crosswalk.csv")
EARS_DIR       = os.path.join(BASE, "Panel", "Intermediate")
CENSUS_CSV     = os.path.join(BASE, "Panel", "census", "acs_tract_timeseries.csv")
GEOID_CSV      = os.path.join(BASE, "Panel", "geo", "case_geoid_lookup.csv")

# ── Step 1: Zoning Case Temporal Anchors ──────────────────────────────────
print("Step 1: Loading zoning cases...")
zoning = pd.read_csv(ZONING_CSV, low_memory=False)

date_cols = ['App_Date', 'Final_Council_Date', 'final_date', 'approval_date', 'application_start_date']
for col in date_cols:
    if col in zoning.columns:
        zoning[col] = pd.to_datetime(zoning[col], errors='coerce')

# Resolve best filing date
zoning['T0'] = zoning['App_Date'].fillna(zoning['application_start_date'])

# Resolve best vote date
zoning['T_vote'] = (
    zoning['Final_Council_Date']
    .fillna(zoning['final_date'])
    .fillna(zoning['approval_date'])
)

# Temporal feature columns
zoning['filing_year']    = zoning['T0'].dt.year
zoning['filing_quarter'] = zoning['T0'].dt.quarter
zoning['vote_year']      = zoning['T_vote'].dt.year
zoning['vote_quarter']   = zoning['T_vote'].dt.quarter
zoning['panel_join_year'] = zoning['filing_year'] - 1

# 4x4 Calendar Quarter Dummies — one for each milestone
def add_quarter_dummies(df, date_col, prefix):
    """Add 4 binary dummy columns (q1-q4) for which calendar quarter a date falls in."""
    q = df[date_col].dt.quarter
    for i in range(1, 5):
        df[f'{prefix}_q{i}'] = (q == i).astype('Int8')
    return df

zoning = add_quarter_dummies(zoning, 'T0',     'filing')
zoning = add_quarter_dummies(zoning, 'T_vote', 'vote')
# Staff and petition quarter dummies added after those joins below

print(f"  {len(zoning)} cases loaded, {zoning['T0'].notna().sum()} with filing date")

# Merge enriched dimensional columns (existing/proposed height, FAR, coverage)
print("  Merging enriched dimensional columns...")
enriched_cols = ['case_number',
                 'existing_max_height_ft','proposed_max_height_ft',
                 'existing_max_far','proposed_max_far',
                 'existing_max_bldg_cov_pct','proposed_max_bldg_cov_pct',
                 'existing_min_lot_sqft','proposed_min_lot_sqft']
try:
    enriched = pd.read_csv(ENRICHED_CSV, low_memory=False, usecols=enriched_cols)
    enriched = enriched.drop_duplicates('case_number')
    zoning = zoning.merge(enriched, on='case_number', how='left')
    print(f"  Enriched columns joined for {enriched['case_number'].nunique()} cases")
except Exception as e:
    print(f"  Enriched join skipped: {e}")

# ── Step 2: Parcel/Tax Join (LDB or EARS depending on year) ───────────────
print("\nStep 2: Loading parcel/tax sources...")

# LDB 2016 (valuation year 2017 — proxy for 2015–2018 filings)
PARCEL_COLS_LDB16 = ['PID_10','MARKET_VAL','APPRAISED_VAL','LAND_ACRES','YR_BUILT','BASEZONE']
ldb16 = pd.read_csv(LDB16_CSV, low_memory=False, usecols=PARCEL_COLS_LDB16)
ldb16 = ldb16.rename(columns={'PID_10':'parcel_id_10','MARKET_VAL':'market_value',
                               'APPRAISED_VAL':'appraised_value','LAND_ACRES':'land_acres',
                               'YR_BUILT':'yr_built','BASEZONE':'basezone'})

def safe_pid(x):
    try:
        return str(int(float(x))).zfill(10)
    except (ValueError, TypeError):
        return None

ldb16['parcel_id_10'] = ldb16['parcel_id_10'].map(safe_pid)
ldb16['data_source'] = 'LDB_2016'

# LDB 2021 (proxy for 2019–2022 filings where EARS unavailable)
PARCEL_COLS_LDB21 = ['PID_10','MARKET_VAL','ASSESSED_V','LAND_ACRES','YR_BUILT','BASEZONE']
ldb21 = pd.read_csv(LDB21_CSV, low_memory=False, usecols=PARCEL_COLS_LDB21)
ldb21 = ldb21.rename(columns={'PID_10':'parcel_id_10','MARKET_VAL':'market_value',
                               'ASSESSED_V':'appraised_value','LAND_ACRES':'land_acres',
                               'YR_BUILT':'yr_built','BASEZONE':'basezone'})
ldb21['parcel_id_10'] = ldb21['parcel_id_10'].map(safe_pid)
ldb21['data_source'] = 'LDB_2021'

# EARS per year (2019–2025 where available) — load on demand
EARS_COLS = ['account_number','total_market_value','appraised_value',
             'land_acres','year_built']

def load_ears_year(year):
    fpath = os.path.join(EARS_DIR, f"ears_{year}_clean.csv")
    if not os.path.exists(fpath):
        return None
    available = pd.read_csv(fpath, nrows=0).columns.tolist()
    load_cols = [c for c in EARS_COLS if c in available]
    df = pd.read_csv(fpath, low_memory=False, usecols=load_cols)
    df = df.rename(columns={'total_market_value': 'market_value', 'year_built': 'yr_built'})
    df['data_source'] = f'EARS_{year}'
    return df

# ID crosswalk: parcel_id_10 → ears_account_number
crosswalk = pd.read_csv(CROSSWALK_CSV, low_memory=False)

# parcel_id_10 already exists in zoning data — just normalize to 10-digit string
zoning['parcel_id_10'] = (
    zoning['parcel_id_10']
    .apply(lambda x: str(int(float(x))).zfill(10) if pd.notna(x) else None)
)
crosswalk['parcel_id_10'] = crosswalk['parcel_id_10'].astype(str).str.zfill(10)
zoning = zoning.merge(crosswalk[['parcel_id_10','ears_account_number']], on='parcel_id_10', how='left')
print(f"  Crosswalk matched: {zoning['ears_account_number'].notna().sum()} / {len(zoning)} cases")

def get_parcel_source(panel_join_year):
    if pd.isna(panel_join_year): return 'LDB_2016'  # pre-2016 fallback
    y = int(panel_join_year)
    if y <= 2018:   return 'LDB_2016'    # covers pre-2016 + 2016-2018
    elif y <= 2020: return 'LDB_2021'
    else:           return f'EARS_{y}'

# Restrict analysis to cases filed 2016+ (expanding window from first LDB vintage)
zoning = zoning[zoning['filing_year'] >= 2016].copy()
print(f"  Restricted to 2016+ filings: {len(zoning)} cases")

zoning['parcel_source'] = zoning['panel_join_year'].apply(get_parcel_source)
print("  Parcel source routing:")
print(zoning['parcel_source'].value_counts().to_string())

# Build a unified parcel lookup per source and join
parcel_rows = []
for source, grp in zoning.groupby('parcel_source'):
    if source == 'none':
        continue
    elif source == 'LDB_2016':
        ref = ldb16
        key = 'parcel_id_10'
        sub = grp[['case_number','parcel_id_10']].dropna()
        merged = sub.merge(ref, on='parcel_id_10', how='left')
    elif source == 'LDB_2021':
        ref = ldb21
        key = 'parcel_id_10'
        sub = grp[['case_number','parcel_id_10']].dropna()
        merged = sub.merge(ref, on='parcel_id_10', how='left')
    elif source.startswith('EARS_'):
        year = int(source.split('_')[1])
        ears = load_ears_year(year)
        if ears is None:
            continue
        sub = grp[['case_number','ears_account_number']].dropna()
        sub['ears_account_number'] = sub['ears_account_number'].astype(str)
        ears['account_number'] = ears['account_number'].astype(str)
        merged = sub.merge(ears, left_on='ears_account_number', right_on='account_number', how='left')
    parcel_rows.append(merged[['case_number','market_value','appraised_value',
                                'land_acres','yr_built']])

if parcel_rows:
    parcel_panel = pd.concat(parcel_rows, ignore_index=True).drop_duplicates('case_number')
    zoning = zoning.merge(parcel_panel, on='case_number', how='left')
    print(f"  Parcel features joined for {parcel_panel['case_number'].nunique()} cases")

# ── Step 3: Protest Petition Join (Endogenous, Calendar-Dated) ────────────
print("\nStep 3: Joining protest petitions...")
petitions = pd.read_csv(PETITION_CSV, low_memory=False)
petitions['date'] = pd.to_datetime(petitions['date'], errors='coerce')
petitions['petition_year']    = petitions['date'].dt.year
petitions['petition_quarter'] = petitions['date'].dt.quarter   # Q1-Q4 calendar

# Aggregate to case level
pet_agg = petitions.groupby('case_number').agg(
    petition_date        = ('date', 'min'),       # earliest signature date
    petition_signer_count = ('signed', 'sum'),    # n signed parcels
    label_petition_total_pct   = ('area_pct', 'sum'),   # total % area signed
    petition_year        = ('petition_year', 'min'),
    petition_quarter     = ('petition_quarter', lambda x: x.iloc[0] if len(x) else None),
).reset_index()

pet_agg['label_valid_protest'] = (pet_agg['label_petition_total_pct'] >= 20).astype(int)

zoning = zoning.merge(pet_agg, on='case_number', how='left')
zoning['label_valid_protest'] = zoning['label_valid_protest'].fillna(0).astype(int)

# Petition quarter dummies (which calendar Q was petition filed?)
zoning['petition_date'] = pd.to_datetime(zoning['petition_date'], errors='coerce')
zoning = add_quarter_dummies(zoning, 'petition_date', 'petition')

# Days-based timing features
zoning['petition_days_from_filing']  = (zoning['petition_date'] - zoning['T0']).dt.days
zoning['petition_days_before_vote']  = (zoning['T_vote'] - zoning['petition_date']).dt.days

print(f"  Valid protests: {zoning['label_valid_protest'].sum()} / {len(zoning)}")

# ── Step 4: Commission/Council Sentiment (Distributed Q-1/Q-2/Q-3 Lags) ──
print("\nStep 4: Computing distributed quarterly sentiment lags...")

# Parse council agendas — use meeting date as sentiment event date per case
council = pd.read_csv(COUNCIL_CSV, low_memory=False)
council['meeting_date'] = pd.to_datetime(council['Meeting_Date'], errors='coerce')

commission = pd.read_csv(COMMISSION_CSV, low_memory=False)
commission['meeting_date'] = pd.to_datetime(commission['Meeting_Date'], errors='coerce')

# Combine, treating each case appearance at a meeting as one sentiment observation
# (For now use appearance count as proxy; replace with VADER when scores available)
council_events = council[['Case_Number','meeting_date']].rename(columns={'Case_Number':'case_number'})
council_events['source'] = 'council'
commission_events = commission[['meeting_date']].copy()  # commission index lacks case_number directly

# For each zoning case, count appearances in each Q-lag window before T0
def count_appearances_in_window(t0, events_df, days_start, days_end):
    """Count event rows strictly within [T0 - days_end, T0 - days_start]."""
    if pd.isna(t0): return 0
    window_start = t0 - pd.Timedelta(days=days_end)
    window_end   = t0 - pd.Timedelta(days=days_start)
    mask = (events_df['meeting_date'] >= window_start) & (events_df['meeting_date'] < window_end)
    return mask.sum()

# Only compute for cases with a filing date
has_t0 = zoning['T0'].notna()
for lag_name, (d_start, d_end) in [('q1', (0, 90)), ('q2', (90, 180)), ('q3', (180, 270))]:
    col = f'council_appearances_{lag_name}'
    zoning.loc[has_t0, col] = zoning.loc[has_t0, 'T0'].apply(
        lambda t0: count_appearances_in_window(t0, council_events, d_start, d_end)
    )
    zoning[col] = zoning[col].fillna(0).astype(int)

print("  Council lag features computed.")

# ── Step 5: Census ACS (Annual, Lag-1) ────────────────────────────────────
print("\nStep 5: Joining census ACS features...")
try:
    census = pd.read_csv(CENSUS_CSV, low_memory=False)
    geoid  = pd.read_csv(GEOID_CSV, low_memory=False)
    census['panel_join_year'] = census['year'].astype(int) - 1 if 'year' in census.columns else None

    zoning = zoning.merge(geoid[['case_number','geoid']], on='case_number', how='left')
    zoning = zoning.merge(
        census[['geoid','panel_join_year','median_household_income','pct_renter_occupied']],
        on=['geoid','panel_join_year'], how='left'
    )
    print("  Census ACS joined.")
except Exception as e:
    print(f"  Census join skipped: {e}")



def milestone_qfeatures(df, date_col, prefix, feature_map):
    """
    For each (q, feature), create column q{q}_{prefix}_{feat} = feature_value
    if the milestone date falls in that calendar quarter, else NaN.
    date_col  : column with the milestone date
    prefix    : e.g. 'filing', 'staff', 'vote', 'petition'
    feature_map: dict of {feat_label: source_col}
    """
    q = df[date_col].dt.quarter  # Series of 1-4 (NaT -> NaN)
    for feat_label, src_col in feature_map.items():
        src = df[src_col] if src_col in df.columns else pd.Series(np.nan, index=df.index)
        for qi in range(1, 5):
            col_name = f'q{qi}_{prefix}_{feat_label}'
            df[col_name] = np.where(q == qi, src, np.nan)
    return df

# Filing milestone features (what developer proposed at filing)
zoning = milestone_qfeatures(zoning, 'T0', 'filing', {
    'height':   'proposed_max_height_ft',
    'far':      'proposed_max_far',
    'coverage': 'proposed_max_bldg_cov_pct',
    'sqft':     'proposed_min_lot_sqft',
})

# Existing baseline features (what was there before the case)
zoning = milestone_qfeatures(zoning, 'T0', 'baseline', {
    'height':   'existing_max_height_ft',
    'far':      'existing_max_far',
    'coverage': 'existing_max_bldg_cov_pct',
    'sqft':     'existing_min_lot_sqft',
})

# Staff milestone features (T_staff currently NaT; populates when date available)
if 'Staff_Report_Date' in zoning.columns:
    zoning['T_staff'] = pd.to_datetime(zoning['Staff_Report_Date'], errors='coerce')
else:
    zoning['T_staff'] = pd.NaT

zoning = milestone_qfeatures(zoning, 'T_staff', 'staff', {
    'height':   'Staff_max_height_ft',
    'far':      'Staff_max_far',
    'coverage': 'Staff_max_bldg_cov_pct',
    'sqft':     'Phase_Staff_SqFt',
})

# Petition milestone features
zoning['petition_date'] = pd.to_datetime(zoning['petition_date'], errors='coerce')
zoning = milestone_qfeatures(zoning, 'petition_date', 'petition', {
    'signer_pct':   'label_petition_total_pct',
    'signer_count': 'petition_signer_count',
})

# Vote milestone features
zoning = milestone_qfeatures(zoning, 'T_vote', 'vote', {
    'height':   'Approved_max_height_ft',
    'far':      'Approved_max_far',
    'coverage': 'Approved_max_bldg_cov_pct',
    'sqft':     'Phase_Approved_SqFt',
})

print("  Milestone feature matrix built.")

# ── Step 7: Assemble Final Panel ──────────────────────────────────────────
print("\nStep 7: Writing final panel...")

# Collect all q{N}_{milestone}_{feature} columns
milestone_cols = [c for c in zoning.columns if c.startswith(('q1_','q2_','q3_','q4_'))]

PARCEL_COLS_OUT = ['market_value','appraised_value','land_acres','yr_built','parcel_source']
CONTEXT_COLS    = ['filing_year','vote_year',
                   'petition_days_from_filing','petition_days_before_vote',
                   'label_valid_protest',
                   'median_household_income','pct_renter_occupied']

OUTCOME_COLS = [
    'label_real_days_in_pipeline',
    'Delta_Approved_Height','Delta_Approved_FAR',
    'Staff_Attrition_Height','Staff_Attrition_SqFt',
    'Remand_Count',
]

all_cols = ['case_number'] + milestone_cols + PARCEL_COLS_OUT + CONTEXT_COLS + OUTCOME_COLS
panel = zoning[[c for c in all_cols if c in zoning.columns]].copy()

out_path = os.path.join(OUT_DIR, 'master_regression_panel.csv')
panel.to_csv(out_path, index=False)
print(f"\nPanel saved: {out_path}")
print(f"Shape: {panel.shape}")
print(f"Milestone columns: {len(milestone_cols)}")
print(f"\nMissingness summary (%):")
print((panel.isnull().mean() * 100).round(1).to_string())
