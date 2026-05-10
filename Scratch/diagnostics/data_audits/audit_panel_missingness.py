"""
audit_panel_missingness.py
Full feature-by-feature missingness audit of the biweekly panel.
For each column: null count, null %, root cause, fixability.
"""
import pandas as pd
import numpy as np
import os

PANEL_PATH  = r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv'
ZONING_PATH = r'c:\Users\dhl\data\Thesis\thesis\Data\model_ready_zoning_data.csv'
PET_PATH    = r'c:\Users\dhl\data\Thesis\thesis\Data\Protest_Petitions\petition_signers_backfilled.csv'

print("Loading panel...")
panel = pd.read_csv(PANEL_PATH, low_memory=False)
panel['period_start'] = pd.to_datetime(panel['period_start'], errors='coerce')

zoning = pd.read_csv(ZONING_PATH, low_memory=False)
pet = pd.read_csv(PET_PATH, low_memory=False)

n_rows  = len(panel)
n_cases = panel['case_number'].nunique()
print(f"Panel: {n_rows:,} rows | {n_cases:,} cases\n")

# Case-level dedup for case-constant column analysis
case_panel = panel.drop_duplicates('case_number').set_index('case_number')

def audit_col(col, df=panel):
    null_n = df[col].isna().sum()
    null_pct = null_n / len(df) * 100
    return null_n, null_pct

print("=" * 80)
print("FEATURE-BY-FEATURE MISSINGNESS AUDIT")
print("=" * 80)

# ── Identifiers & Time ─────────────────────────────────────────────────────
print("\n--- IDENTIFIERS & TIME ---")
for col in ['case_number', 'period_start', 'period_seq', 'year', 'quarter']:
    if col in panel.columns:
        n, p = audit_col(col)
        print(f"  {col:35s}  null={n:6,}  ({p:5.1f}%)")
        if col == 'case_number' and n > 0:
            print(f"    >> {n} rows have no case_number — likely skeleton rows from NaN cases in zoning CSV")
        if col == 'period_start' and n > 0:
            print(f"    >> period_start null means T0 was NaT — should not happen after T0.notna() filter")

# ── Event Flags ────────────────────────────────────────────────────────────
print("\n--- EVENT FLAGS ---")
for col in ['filing_event', 'petition_event', 'vote_event', 'resolved']:
    if col in panel.columns:
        n, p = audit_col(col)
        n1 = int(panel[col].sum()) if col in panel.columns else 0
        print(f"  {col:35s}  null={n:6,}  ({p:5.1f}%)  |  fires={n1:,} rows")
        if col == 'petition_event':
            # Check how many cases have a petition in the source
            n_with_petition = pet['case_number'].nunique()
            pet['date_parsed'] = pd.to_datetime(pet['date'], format='mixed', errors='coerce')
            n_parsed = (pet['date_parsed'].notna()).sum()
            print(f"    >> Petition source: {len(pet):,} rows, {n_with_petition} cases")
            print(f"    >> Dates parseable: {n_parsed}/{len(pet)} ({n_parsed/len(pet)*100:.1f}%)")
            print(f"    >> Root cause: pd.to_datetime(format='mixed') required — 'July 29, 2008' format not auto-detected")
        if col == 'vote_event':
            n_no_vote = int((zoning['Final_Council_Date'].isna() & zoning['final_date'].isna() & zoning['approval_date'].isna()).sum())
            print(f"    >> {n_no_vote} cases in zoning have no vote date at all (genuinely unresolved)")

# ── Dimensional Features at Event Rows ─────────────────────────────────────
print("\n--- DIMENSIONAL FEATURES AT EVENT ROWS ---")
enr_path = r'c:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\enriched_zoning_data_causal.csv'
enr = pd.read_csv(enr_path, low_memory=False)
enr_cases = enr['case_number'].nunique()
for col in ['height_at_event', 'far_at_event', 'coverage_at_event']:
    if col in panel.columns:
        n, p = audit_col(col)
        # Non-null should only exist in filing/vote rows
        filing_rows   = panel[panel['filing_event'] == 1]
        vote_rows     = panel[panel['vote_event'] == 1]
        filing_nonnull = filing_rows[col].notna().sum()
        vote_nonnull   = vote_rows[col].notna().sum()
        print(f"  {col:35s}  null={n:6,}  ({p:5.1f}%)")
        print(f"    >> Filing rows with value: {filing_nonnull}/{len(filing_rows)} | Vote rows with value: {vote_nonnull}/{len(vote_rows)}")
        print(f"    >> Root cause: enriched_zoning_data_causal.csv only covers {enr_cases} cases")
        print(f"    >> Fix: re-run calculate_zoning_metrics.py against full 6,757 case set")

# ── Petition Features ──────────────────────────────────────────────────────
print("\n--- PETITION FEATURES ---")
for col in ['petition_pct_this_period', 'petition_count_this_period', 'label_valid_protest']:
    if col in panel.columns:
        n, p = audit_col(col)
        print(f"  {col:35s}  null={n:6,}  ({p:5.1f}%)")
        if col == 'petition_pct_this_period':
            print(f"    >> Root cause: date parsing — 'July 29, 2008' format needs format='mixed'")
            print(f"    >> Also: area_pct is 0 for 75% of signers (unsigned parcels) — only {(pd.read_csv(PET_PATH)['area_pct']>0).sum()} rows have area_pct>0")
        if col == 'label_valid_protest':
            n_vp = int(panel['label_valid_protest'].sum()) if 'label_valid_protest' in panel.columns else 0
            print(f"    >> label_valid_protest=1 in {n_vp} rows across {panel[panel['label_valid_protest']==1]['case_number'].nunique() if 'label_valid_protest' in panel.columns else 0} cases")

# ── Council Hearings ───────────────────────────────────────────────────────
print("\n--- COUNCIL HEARINGS ---")
if 'council_hearings_this_period' in panel.columns:
    n, p = audit_col('council_hearings_this_period')
    n_active = int((panel['council_hearings_this_period'] > 0).sum())
    print(f"  {'council_hearings_this_period':35s}  null={n:6,}  ({p:5.1f}%)  |  periods_with_hearing={n_active:,}")
    council_path = r'c:\Users\dhl\data\Thesis\thesis\Data\council_agendas_cases.csv'
    cdf = pd.read_csv(council_path)
    print(f"    >> Council source: {len(cdf):,} rows, {cdf['Case_Number'].nunique()} cases, years: {cdf['Year'].min()}-{cdf['Year'].max()}")
    in_panel = cdf['Case_Number'].isin(panel['case_number'])
    print(f"    >> Cases from council agenda matched in panel: {in_panel.sum()} / {len(cdf)}")

# ── Parcel Features (Forward-Filled Annual) ────────────────────────────────
print("\n--- PARCEL FEATURES (FORWARD-FILLED ANNUAL) ---")
for col in ['market_value', 'appraised_value', 'land_acres', 'yr_built']:
    if col in panel.columns:
        n, p = audit_col(col)
        # Case-level: how many unique cases have any parcel data?
        cases_with_val = panel.dropna(subset=[col])['case_number'].nunique()
        print(f"  {col:35s}  null={n:6,}  ({p:5.1f}%)  |  cases_with_data={cases_with_val:,}/{n_cases:,}")

# Diagnose the parcel gap
print()
zoning_ids = pd.read_csv(ZONING_PATH, low_memory=False, usecols=['case_number','parcel_id_10'])
no_pid = zoning_ids['parcel_id_10'].isna().sum()
has_pid = zoning_ids['parcel_id_10'].notna().sum()
crosswalk = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\Panel\Reference\id_crosswalk.csv')
crosswalk['parcel_id_10'] = crosswalk['parcel_id_10'].astype(str).str.zfill(10)
zoning_ids['pid_str'] = zoning_ids['parcel_id_10'].apply(
    lambda x: str(int(float(x))).zfill(10) if pd.notna(x) else None)
matched = zoning_ids['pid_str'].isin(crosswalk['parcel_id_10']).sum()
print(f"  Parcel gap breakdown:")
print(f"    {no_pid:,} cases have no parcel_id_10 in zoning CSV  -> must use LDB direct join on parcel_id_10")
print(f"    {has_pid - matched:,} cases have parcel_id_10 but no EARS crosswalk match  -> covered by LDB fallback")
print(f"    {matched:,} cases fully crosswalk-matched to EARS account")
print(f"    Root cause of remaining 50% null: LDB join is on parcel_id_10 but many cases")
print(f"    have parcel_id_10 in zoning yet aren't yet routed through the LDB direct path")

# ── Spatial ────────────────────────────────────────────────────────────────
print("\n--- SPATIAL FEATURES ---")
for col in ['latitude', 'longitude', 'shape_area', 'council_district']:
    if col in panel.columns:
        n, p = audit_col(col)
        cases_with_val = panel.dropna(subset=[col])['case_number'].nunique()
        print(f"  {col:35s}  null={n:6,}  ({p:5.1f}%)  |  cases_with_data={cases_with_val:,}/{n_cases:,}")
        if col == 'latitude' and p > 0:
            # Check source
            z_lat = zoning_ids.merge(
                pd.read_csv(ZONING_PATH, low_memory=False, usecols=['case_number','latitude']),
                on='case_number', how='left')
            print(f"    >> In zoning source: {pd.read_csv(ZONING_PATH, low_memory=False, usecols=['latitude'])['latitude'].notna().sum()} / {len(zoning_ids)} have lat")

# ── Outcome / Target Variables ─────────────────────────────────────────────
print("\n--- OUTCOME VARIABLES ---")
for col in ['label_real_days_in_pipeline', 'Remand_Count', 'Delta_Approved_Height',
            'Delta_Approved_FAR', 'Staff_Attrition_Height',
            'Aggregate_Sentiment', 'label_valid_petition_pct']:
    if col in panel.columns:
        n, p = audit_col(col)
        cases_null = panel[panel[col].isna()]['case_number'].nunique()
        print(f"  {col:35s}  null={n:6,}  ({p:5.1f}%)  |  null_cases={cases_null:,}")
        if col == 'label_real_days_in_pipeline':
            # Check status breakdown of nulls
            null_cases = panel[panel[col].isna()]['case_number'].unique()
            z_status = pd.read_csv(ZONING_PATH, low_memory=False, usecols=['case_number','Derived_Status'])
            null_statuses = z_status[z_status['case_number'].isin(null_cases)]['Derived_Status'].value_counts()
            print(f"    >> Null case statuses:")
            for status, cnt in null_statuses.head(5).items():
                print(f"       {cnt:5,}  {status}")
            print(f"    >> Root cause: unresolved cases (no final vote) → right-censored, not missing")
            print(f"    >> Fix: set label_real_days_in_pipeline = days_to_cutoff (2026-05-02), censored=1 flag")
        if col == 'Aggregate_Sentiment':
            print(f"    >> Root cause: sentiment extracted only for cases with matching transcripts/agendas")
        if col == 'Delta_Approved_Height':
            print(f"    >> Root cause: requires enriched file (same gap as height_at_event — 244 cases only)")

print("\n=== SUMMARY ===")
print(f"Total panel rows:  {n_rows:,}")
print(f"Total cases:       {n_cases:,}")
miss = (panel.isnull().mean() * 100).round(1).sort_values(ascending=False)
print("\nAll columns by missingness:")
for col, pct in miss.items():
    bar = '█' * int(pct / 5)
    print(f"  {col:35s}  {pct:5.1f}%  {bar}")
