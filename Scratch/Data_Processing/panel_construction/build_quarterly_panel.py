"""
build_quarterly_panel.py
Builds a quarterly panel: one row per (case_number, year, quarter).
- Annual static data (EARS/LDB) is forward-filled across each quarter of the year.
- Pipeline milestone events (filing, petition, vote) are flagged in the Q they occur.
- All 6,757 zoning cases included; pre-2016 cases use LDB_2016 as proxy.
- Iteratively run: add data sources to expand coverage.
"""
import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = r"c:\Users\dhl\data\Thesis\thesis\Data"
OUT_DIR = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756"

ZONING_CSV     = os.path.join(BASE, "model_ready_zoning_data.csv")
ENRICHED_CSV   = os.path.join(BASE, "Zoning_Cases", "Processed_Data", "CSV", "enriched_zoning_data_causal.csv")
PETITION_CSV   = os.path.join(BASE, "Protest_Petitions", "petition_signers_backfilled.csv")
COUNCIL_CSV    = os.path.join(BASE, "council_agendas_cases.csv")
LDB16_CSV      = os.path.join(BASE, "CoA_Open_Data", "LDB_2016_4nsn-uea6.csv")
LDB21_CSV      = os.path.join(BASE, "CoA_Open_Data", "LDB_2021_kk8y-6cmt.csv")
CROSSWALK_CSV  = os.path.join(BASE, "Panel", "Reference", "id_crosswalk.csv")
EARS_DIR       = os.path.join(BASE, "Panel", "Intermediate")

def safe_pid(x):
    try:
        return str(int(float(x))).zfill(10)
    except (ValueError, TypeError):
        return None

# ── Step 1: Load Zoning Cases & Resolve Dates ──────────────────────────────
print("Step 1: Loading zoning cases...")
zoning = pd.read_csv(ZONING_CSV, low_memory=False)

for col in ['App_Date', 'Final_Council_Date', 'final_date', 'approval_date']:
    if col in zoning.columns:
        zoning[col] = pd.to_datetime(zoning[col], errors='coerce')

zoning['T0']     = zoning['App_Date']
zoning['T_vote'] = zoning['Final_Council_Date'].fillna(zoning['final_date']).fillna(zoning['approval_date'])

# All cases — no year restriction
# For cases without a T0, drop (can't place in time)
zoning = zoning[zoning['T0'].notna()].copy()

zoning['filing_year']    = zoning['T0'].dt.year
zoning['filing_quarter'] = zoning['T0'].dt.quarter
zoning['vote_year']      = zoning['T_vote'].dt.year
zoning['vote_quarter']   = zoning['T_vote'].dt.quarter

print(f"  {len(zoning)} cases with a valid filing date")
print(f"  Filing year range: {int(zoning['filing_year'].min())} – {int(zoning['filing_year'].max())}")

# ── Step 2: Build Quarterly Skeleton ──────────────────────────────────────
# For each case, generate one row per (year, quarter) it was active.
# Active = from filing quarter to vote quarter (or filing quarter only if no vote).
print("\nStep 2: Building quarterly skeleton...")

rows = []
for _, row in zoning.iterrows():
    t0 = row['T0']
    tv = row['T_vote'] if pd.notna(row['T_vote']) else t0 + pd.DateOffset(days=365)

    # Generate quarterly periods covering pipeline duration
    periods = pd.period_range(
        start=pd.Period(t0, freq='Q'),
        end=pd.Period(tv, freq='Q'),
        freq='Q'
    )
    for p in periods:
        rows.append({
            'case_number': row['case_number'],
            'year':        p.year,
            'quarter':     p.quarter,
            # milestone flags — will be 1 in the Q the event occurred
            'filing_event':   int(p.year == t0.year and p.quarter == t0.quarter),
            'vote_event':     int(pd.notna(row['T_vote']) and
                                  p.year == row['T_vote'].year and
                                  p.quarter == row['T_vote'].quarter),
        })

panel = pd.DataFrame(rows)
print(f"  Quarterly panel: {len(panel)} rows ({panel['case_number'].nunique()} cases)")

# Merge case-level columns back onto panel
case_cols = ['case_number', 'filing_year', 'filing_quarter', 'vote_year', 'vote_quarter',
             'label_real_days_in_pipeline', 'Remand_Count', 'Council_Appearances',
             'Aggregate_Sentiment', 'label_valid_petition_pct',
             'Delta_Approved_Height', 'Delta_Approved_FAR',
             'Staff_Attrition_Height', 'Remand_Count',
             'parcel_id_10', 'latitude', 'longitude', 'shape_area',
             'council_district', 'T0', 'T_vote']
case_meta = zoning[[c for c in case_cols if c in zoning.columns]].drop_duplicates('case_number')
panel = panel.merge(case_meta, on='case_number', how='left')

# ── Step 3: Annual Parcel Data → Forward-Fill onto Quarters ───────────────
print("\nStep 3: Joining annual parcel data (forward-filled to quarters)...")

# Load parcel sources
LDB_COLS_16 = ['PID_10','MARKET_VAL','APPRAISED_VAL','LAND_ACRES','YR_BUILT']
ldb16 = pd.read_csv(LDB16_CSV, low_memory=False, usecols=LDB_COLS_16)
ldb16 = ldb16.rename(columns={'PID_10':'parcel_id_10','MARKET_VAL':'market_value',
                               'APPRAISED_VAL':'appraised_value','LAND_ACRES':'land_acres',
                               'YR_BUILT':'yr_built'})
ldb16['parcel_id_10'] = ldb16['parcel_id_10'].map(safe_pid)
ldb16['data_year'] = 2016

LDB_COLS_21 = ['PID_10','MARKET_VAL','ASSESSED_V','LAND_ACRES','YR_BUILT']
ldb21 = pd.read_csv(LDB21_CSV, low_memory=False, usecols=LDB_COLS_21)
ldb21 = ldb21.rename(columns={'PID_10':'parcel_id_10','MARKET_VAL':'market_value',
                               'ASSESSED_V':'appraised_value','LAND_ACRES':'land_acres',
                               'YR_BUILT':'yr_built'})
ldb21['parcel_id_10'] = ldb21['parcel_id_10'].map(safe_pid)
ldb21['data_year'] = 2021

# EARS annual files
EARS_COLS = ['account_number','total_market_value','appraised_value','land_acres','year_built']
ears_frames = []
for yr in range(2019, 2026):
    fpath = os.path.join(EARS_DIR, f"ears_{yr}_clean.csv")
    if not os.path.exists(fpath):
        continue
    avail = pd.read_csv(fpath, nrows=0).columns.tolist()
    cols = [c for c in EARS_COLS if c in avail]
    df = pd.read_csv(fpath, low_memory=False, usecols=cols)
    df = df.rename(columns={'total_market_value':'market_value','year_built':'yr_built'})
    df['data_year'] = yr
    ears_frames.append(df)

# Normalize parcel_id_10 from zoning data
panel['parcel_id_10'] = panel['parcel_id_10'].map(safe_pid)
crosswalk = pd.read_csv(CROSSWALK_CSV, low_memory=False)
crosswalk['parcel_id_10'] = crosswalk['parcel_id_10'].astype(str).str.zfill(10)
panel = panel.merge(crosswalk[['parcel_id_10','ears_account_number']], on='parcel_id_10', how='left')

def best_parcel_data_for_year(year):
    """Return parcel snapshot closest to (but not exceeding) the given year."""
    if year <= 2018:
        return ldb16  # use LDB 2016 for all pre-2019
    elif year <= 2020:
        return ldb21
    else:
        # Use EARS for exact year or nearest available
        target = min(year, 2025)
        for yr in range(target, 2018, -1):
            matches = [f for f in ears_frames if f['data_year'].iloc[0] == yr]
            if matches:
                return matches[0]
        return ldb21

PARCEL_FEATURES = ['market_value','appraised_value','land_acres','yr_built']

# For each unique (parcel_id_10, year) combo in the panel, join the right parcel snapshot
# This is the forward-fill: same annual value applies to all 4 quarters of that year
panel_years = panel[['case_number','parcel_id_10','ears_account_number','year']].drop_duplicates()
parcel_joined = []

for yr, grp in panel_years.groupby('year'):
    ref = best_parcel_data_for_year(yr)
    if 'account_number' in ref.columns:
        # EARS join via account number
        sub = grp.dropna(subset=['ears_account_number'])
        sub = sub.copy()
        sub['ears_account_number'] = sub['ears_account_number'].astype(str)
        ref['account_number'] = ref['account_number'].astype(str)
        m = sub.merge(ref[['account_number'] + PARCEL_FEATURES],
                      left_on='ears_account_number', right_on='account_number', how='left')
    else:
        # LDB join via parcel_id_10
        sub = grp.dropna(subset=['parcel_id_10'])
        m = sub.merge(ref[['parcel_id_10'] + PARCEL_FEATURES], on='parcel_id_10', how='left')
    parcel_joined.append(m[['case_number','year'] + PARCEL_FEATURES])

if parcel_joined:
    pj = pd.concat(parcel_joined, ignore_index=True).drop_duplicates(['case_number','year'])
    panel = panel.merge(pj, on=['case_number','year'], how='left')
    print(f"  Parcel data joined for {pj['case_number'].nunique()} unique case-year combos")

# ── Step 4: Protest Petition Events ───────────────────────────────────────
print("\nStep 4: Adding protest petition events...")
petitions = pd.read_csv(PETITION_CSV, low_memory=False)
petitions['date'] = pd.to_datetime(petitions['date'], errors='coerce')

pet_agg = petitions.groupby('case_number').agg(
    petition_date         = ('date', 'min'),
    petition_signer_count = ('signed', 'sum'),
    label_petition_total_pct    = ('area_pct', 'sum'),
).reset_index()
pet_agg['label_valid_protest']      = (pet_agg['label_petition_total_pct'] >= 20).astype(int)
pet_agg['petition_year']      = pet_agg['petition_date'].dt.year
pet_agg['petition_quarter']   = pet_agg['petition_date'].dt.quarter

panel = panel.merge(pet_agg, on='case_number', how='left')
panel['label_valid_protest'] = panel['label_valid_protest'].fillna(0).astype(int)

# Flag rows where this is the quarter the petition was filed
panel['petition_event'] = (
    (panel['year'] == panel['petition_year']) &
    (panel['quarter'] == panel['petition_quarter'])
).astype(int)

# Petition feature values only visible in the Q the petition was filed
panel['petition_signer_pct_this_q']   = np.where(panel['petition_event'] == 1,
                                                   panel['label_petition_total_pct'], np.nan)
panel['petition_signer_count_this_q'] = np.where(panel['petition_event'] == 1,
                                                   panel['petition_signer_count'], np.nan)
print(f"  Petition events: {panel['petition_event'].sum()} quarters with a petition filing")

# ── Dimensional values at each event row ──────────────────────────────────
# Join enriched proposed/existing height+FAR columns from enriched file
try:
    enriched = pd.read_csv(ENRICHED_CSV, low_memory=False,
                           usecols=['case_number','proposed_max_height_ft','proposed_max_far',
                                    'existing_max_height_ft','existing_max_far',
                                    'proposed_max_bldg_cov_pct','existing_max_bldg_cov_pct'])
    enriched = enriched.drop_duplicates('case_number')
    panel = panel.merge(enriched, on='case_number', how='left')
    print(f"  Enriched dimensional columns joined for {enriched['case_number'].nunique()} cases")
except Exception as e:
    print(f"  Enriched join skipped: {e}")
    for c in ['proposed_max_height_ft','proposed_max_far','existing_max_height_ft',
              'existing_max_far','proposed_max_bldg_cov_pct','existing_max_bldg_cov_pct']:
        panel[c] = np.nan

# height_at_event: proposed height in filing row, approved (existing before vote) in vote row, null otherwise
panel['height_at_event'] = np.where(panel['filing_event'] == 1, panel['proposed_max_height_ft'],
                           np.where(panel['vote_event'] == 1,   panel['existing_max_height_ft'],
                                                                np.nan))
panel['far_at_event']    = np.where(panel['filing_event'] == 1, panel['proposed_max_far'],
                           np.where(panel['vote_event'] == 1,   panel['existing_max_far'],
                                                                np.nan))
panel['coverage_at_event'] = np.where(panel['filing_event'] == 1, panel['proposed_max_bldg_cov_pct'],
                              np.where(panel['vote_event'] == 1,   panel['existing_max_bldg_cov_pct'],
                                                                    np.nan))

# ── Step 5: Council Appearance Events ─────────────────────────────────────
print("\nStep 5: Adding council appearance events...")
council = pd.read_csv(COUNCIL_CSV, low_memory=False)
council['meeting_date'] = pd.to_datetime(council['Meeting_Date'], errors='coerce')
council['year']    = council['meeting_date'].dt.year
council['quarter'] = council['meeting_date'].dt.quarter

council_agg = council.groupby(['Case_Number','year','quarter']).size().reset_index(name='council_appearances_this_q')
council_agg = council_agg.rename(columns={'Case_Number':'case_number'})
panel = panel.merge(council_agg, on=['case_number','year','quarter'], how='left')
panel['council_appearances_this_q'] = panel['council_appearances_this_q'].fillna(0).astype(int)

# ── Step 6: Write Output ───────────────────────────────────────────────────
print("\nStep 6: Writing quarterly panel...")

# Outcome columns are case-level (constant across all quarters for a case)
# Predictor columns are quarterly (vary by quarter)
OUTPUT_COLS = [
    'case_number', 'year', 'quarter',
    # Milestone flags
    'filing_event', 'vote_event', 'petition_event',
    # Filing metadata (constant for case)
    'filing_year', 'filing_quarter', 'vote_year', 'vote_quarter',
    'council_district',
    # Quarterly parcel data (forward-filled from annual)
    'market_value', 'appraised_value', 'land_acres', 'yr_built',
    # Quarterly council activity
    'council_appearances_this_q',
    # Dimensional values at event rows
    'height_at_event', 'far_at_event', 'coverage_at_event',
    # Petition features (non-null only in petition Q)
    'petition_event', 'petition_signer_pct_this_q', 'petition_signer_count_this_q',
    'label_valid_protest', 'petition_quarter',
    # Spatial
    'latitude', 'longitude', 'shape_area',
    # Case-level outcomes (for regression)
    'label_real_days_in_pipeline', 'Remand_Count',
    'Delta_Approved_Height', 'Delta_Approved_FAR',
    'Staff_Attrition_Height',
    'Aggregate_Sentiment', 'label_valid_petition_pct',
]

out = panel[[c for c in OUTPUT_COLS if c in panel.columns]].copy()
out_path = os.path.join(OUT_DIR, 'quarterly_regression_panel.csv')
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {out.shape}")
print(f"Cases covered: {out['case_number'].nunique()}")
print(f"\nMissingness summary (%):")
print((out.isnull().mean() * 100).round(1).sort_values(ascending=False).head(20).to_string())
