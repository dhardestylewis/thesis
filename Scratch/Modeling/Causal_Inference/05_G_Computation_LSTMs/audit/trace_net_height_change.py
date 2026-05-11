"""
trace_net_height_change.py
Trace the exact construction of net_height_change for every nonzero case.
Answer: are these real observed reductions or LDC arithmetic artifacts?
"""
import pandas as pd
import numpy as np

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv', low_memory=False)
pdf_ht = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\interim\pdf_height_features.csv', low_memory=False)

nhc = pd.to_numeric(df['net_height_change'], errors='coerce')
nonzero_cases = df.loc[nhc > 0, 'case_number'].unique()
print(f"Cases with net_height_change > 0: {len(nonzero_cases)}")
print()

# For each nonzero case, pull the components that made it nonzero
# Formula from 01d_merge_pdf_height_features.py line 108-112:
#   initial_req = groupby(case).pdf_requested_height_ft.transform('max')
#   current_constraint = min(pdf_requested_height_ft, pdf_staff_recommends_ht)
#   final_ht = pdf_reduced_to_ft.fillna(current_constraint).fillna(0)
#   net_height_change = (initial_req - final_ht).clip(0)

# So net_height_change > 0 means: initial_req > final_ht
# Which is true when:
#   A) pdf_reduced_to_ft < initial_req (explicit stated reduction)
#   B) pdf_staff_recommends_ht < pdf_requested_height_ft (staff recommends lower than request)
#   C) pdf_requested_height_ft itself drops between periods (if it varies per row)

# Pull the last nonzero row for each affected case
sub = df[df['case_number'].isin(nonzero_cases)].copy()
sub['nhc'] = pd.to_numeric(sub['net_height_change'], errors='coerce')
sub['req_ht'] = pd.to_numeric(sub['pdf_requested_height_ft'], errors='coerce')
sub['red_ht'] = pd.to_numeric(sub['pdf_reduced_to_ft'], errors='coerce')
sub['staff_ht'] = pd.to_numeric(sub['pdf_staff_recommends_ht'], errors='coerce')
sub['compat_ht'] = pd.to_numeric(sub['pdf_compatibility_height_ft'], errors='coerce')
sub['petition'] = pd.to_numeric(sub['petition_pct_this_period'], errors='coerce').fillna(0)
sub['cum_petition'] = pd.to_numeric(sub['cumulative_petition_pct'], errors='coerce').fillna(0)

# Get per-case summary (max nonzero nhc row)
case_rows = (sub[sub['nhc'] > 0]
             .sort_values('nhc', ascending=False)
             .drop_duplicates('case_number'))

# Classify mechanism for each case
def classify(row):
    if pd.notna(row['red_ht']) and row['red_ht'] > 0:
        return 'A_explicit_reduced_to_ft'
    elif pd.notna(row['staff_ht']) and pd.notna(row['req_ht']) and row['staff_ht'] < row['req_ht']:
        return 'B_staff_recommends_lower'
    elif pd.notna(row['compat_ht']) and pd.notna(row['req_ht']) and row['compat_ht'] < row['req_ht']:
        return 'C_compat_constraint'
    elif pd.notna(row['req_ht']):
        return 'D_req_vs_initial_ldc_drop'
    return 'E_unknown'

case_rows['mechanism'] = case_rows.apply(classify, axis=1)

print("=== MECHANISM BREAKDOWN (what actually produces net_height_change > 0) ===")
print(case_rows['mechanism'].value_counts().to_string())
print()

print("=== PETITION ACTIVITY among nonzero net_height_change cases ===")
case_petition = sub.groupby('case_number')['petition'].max()
print(f"  Cases with any petition (max dose > 0): {(case_petition > 0).sum()} / {len(nonzero_cases)}")
print(f"  Cases with substantial petition (> 0.01): {(case_petition > 0.01).sum()} / {len(nonzero_cases)}")
print()

print("=== PER-MECHANISM: petition overlap ===")
for mech, grp in case_rows.groupby('mechanism'):
    pet_cases = (grp['petition'] > 0).sum()
    print(f"  {mech:<35} n={len(grp):>3}  petitioned={pet_cases:>3}  "
          f"mean_nhc={grp['nhc'].mean():.1f}ft  mean_petition={grp['petition'].mean():.4f}")

print()
print("=== MECHANISM D: LDC arithmetic deep-dive ===")
d_cases = case_rows[case_rows['mechanism'] == 'D_req_vs_initial_ldc_drop']
if len(d_cases) > 0:
    # These are the suspicious ones — no explicit staff or reduced_to signal
    # net_height_change fires because pdf_requested_height_ft varies across periods
    # (merge_asof brings in different source_date values at different periods)
    print(f"  Cases in D: {len(d_cases)}")
    print(f"  These have req_ht > 0 but no explicit reduction signal")
    # Check if req_ht varies per period for these cases
    for case in d_cases['case_number'].head(5):
        case_df = sub[sub['case_number'] == case]
        req_vals = case_df['req_ht'].dropna().unique()
        print(f"    {case}: req_ht values = {sorted(req_vals)}  nhc_range = {case_df['nhc'].min():.0f}-{case_df['nhc'].max():.0f}")
    print()

print("=== MECHANISM A: Explicit pdf_reduced_to_ft cases with petition ===")
a_cases = case_rows[case_rows['mechanism'] == 'A_explicit_reduced_to_ft']
print(f"  Total A cases: {len(a_cases)}")
pet_a = a_cases[a_cases['petition'] > 0]
print(f"  Of these, petitioned: {len(pet_a)}")
if len(a_cases) > 0:
    print(a_cases[['case_number','req_ht','red_ht','nhc','petition','cum_petition']].to_string(index=False))
print()

print("=== MECHANISM B: staff_recommends_ht < requested with petition ===")
b_cases = case_rows[case_rows['mechanism'] == 'B_staff_recommends_lower']
print(f"  Total B cases: {len(b_cases)}")
pet_b = b_cases[b_cases['petition'] > 0]
print(f"  Of these, petitioned: {len(pet_b)}")
if len(b_cases) > 0:
    print(b_cases[['case_number','req_ht','staff_ht','nhc','petition','cum_petition']].to_string(index=False))
print()

print("=== CROSS-CHECK against pdf_height_features.csv (raw extraction) ===")
# How many of the 76 cases actually appear in the raw PDF extraction?
pdf_cases = set(pdf_ht['case_number'].unique())
overlap = [c for c in nonzero_cases if c in pdf_cases]
print(f"  Cases in nonzero nhc that appear in pdf_height_features: {len(overlap)} / {len(nonzero_cases)}")
print(f"  Cases with nhc > 0 but NO pdf extraction: {len(nonzero_cases) - len(overlap)}")
print()

# The ones with nhc > 0 but no PDF extraction are pure LDC table arithmetic
no_pdf = [c for c in nonzero_cases if c not in pdf_cases]
if no_pdf:
    print(f"  Cases with net_height_change > 0 from LDC arithmetic ONLY (no PDF): {len(no_pdf)}")
    for c in no_pdf[:10]:
        row = sub[sub['case_number'] == c].iloc[0]
        print(f"    {c}: nhc={row['nhc']:.0f}  req_ht={row['req_ht']}  petition={row['petition']:.4f}")
