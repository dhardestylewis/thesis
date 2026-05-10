"""
What is associated with protest? 
Bivariate comparison of protested vs non-protested cases across all key dimensions.
Uses case-level snapshots (first period for architectural/economic features,
max for process features).
"""
import pandas as pd
import numpy as np
from scipy import stats

PANEL_PATH  = r"C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv"
MASTER_PATH = r"C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv"

panel  = pd.read_csv(PANEL_PATH,  low_memory=False)
master = pd.read_csv(MASTER_PATH, low_memory=False)

panel['period_start'] = pd.to_datetime(panel['period_start'], errors='coerce')

# ── Case-level protest flag ──────────────────────────────────────────────────
protested_cases = set(panel[panel['petition_event'] == 1]['case_number'].unique())

# ── Build case-level summary ─────────────────────────────────────────────────
# First period = filing-time features (no leakage)
first = (panel.sort_values('period_seq')
         .groupby('case_number').first().reset_index())

# Max over lifecycle = process totals
maxp = (panel.groupby('case_number').agg(
    total_periods=('period_seq','max'),
    total_council_hearings=('cumulative_council_hearings','max'),
    total_commission_hearings=('cumulative_commission_hearings','max'),
    total_remands=('Remand_Count','max'),
    max_knn_petition_rate=('knn_petition_rate_1km','max'),
    max_dist_petition_rate=('dist_petition_rate_lag1','max'),
).reset_index())

# Join master for case_type, commission_type
master_sub = master[['case_number','case_type','Commission_Type',
                      'Opposition_Volume','Support_Volume']].drop_duplicates('case_number')

case = first.merge(maxp, on='case_number', how='left')
case = case.merge(master_sub, on='case_number', how='left')
case['protested'] = case['case_number'].isin(protested_cases)

print(f"Total cases: {len(case):,}  |  Protested: {case['protested'].sum():,} ({case['protested'].mean()*100:.1f}%)\n")

# ── Numeric bivariate comparison ─────────────────────────────────────────────
num_features = [
    ('proposed_max_height_ft',    'Proposed height (ft)'),
    ('existing_max_height_ft',    'Existing height (ft)'),
    ('height_delta',              'Height delta (ft)'),
    ('proposed_max_far',          'Proposed FAR'),
    ('existing_max_far',          'Existing FAR'),
    ('land_acres',                'Site area (acres)'),
    ('market_value',              'Market value ($k)'),
    ('median_household_income',   'Median HH income'),
    ('renter_share',              'Renter share'),
    ('rent_burden',               'Rent burden'),
    ('total_population',          'Population (tract)'),
    ('knn_petition_rate_1km',     'KNN petition rate 1km (at filing)'),
    ('dist_petition_rate_lag1',   'Dist petition rate lag1 (at filing)'),
    ('max_knn_petition_rate',     'KNN petition rate 1km (max over case)'),
    ('total_council_hearings',    'Total council hearings'),
    ('total_commission_hearings', 'Total commission hearings'),
    ('total_remands',             'Total remands'),
    ('total_periods',             'Total periods in panel'),
    ('Opposition_Volume',         'Opposition volume (council)'),
]

rows = []
for col, label in num_features:
    if col not in case.columns:
        continue
    p_grp  = case[case['protested']][col].dropna()
    np_grp = case[~case['protested']][col].dropna()
    if len(p_grp) < 5 or len(np_grp) < 5:
        continue
    _, pval = stats.mannwhitneyu(p_grp, np_grp, alternative='two-sided')
    rows.append({
        'Feature': label,
        'Non-protested median': np_grp.median(),
        'Protested median':     p_grp.median(),
        'Ratio':                p_grp.median() / (np_grp.median() + 1e-9),
        'p-value':              pval,
    })

results = pd.DataFrame(rows).sort_values('p-value')
results['sig'] = results['p-value'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_colwidth', 40)
print("=== Numeric features: median comparison (Mann-Whitney U) ===")
print(results.to_string(index=False))

# ── Categorical: council district ────────────────────────────────────────────
print("\n=== Protest rate by council district ===")
dist_rate = (case.groupby('council_district')['protested']
             .agg(['mean','sum','count'])
             .rename(columns={'mean':'protest_rate','sum':'n_protested','count':'n_total'})
             .sort_values('protest_rate', ascending=False)
             .head(12))
dist_rate['protest_rate'] = (dist_rate['protest_rate']*100).round(1)
print(dist_rate.to_string())

# ── Categorical: case_type ───────────────────────────────────────────────────
print("\n=== Protest rate by case_type ===")
ct_rate = (case.groupby('case_type')['protested']
           .agg(['mean','sum','count'])
           .rename(columns={'mean':'protest_rate','sum':'n_protested','count':'n_total'})
           .sort_values('protest_rate', ascending=False))
ct_rate['protest_rate'] = (ct_rate['protest_rate']*100).round(1)
print(ct_rate.to_string())
