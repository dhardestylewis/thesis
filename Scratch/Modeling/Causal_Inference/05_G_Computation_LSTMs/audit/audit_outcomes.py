import pandas as pd, numpy as np

df = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\Panel\biweekly_panel.csv', low_memory=False)
print('Panel shape:', df.shape)
print()

outcome_cols = ['net_height_change','Delta_Approved_Height','Staff_Attrition_Height',
                'pdf_reduced_to_ft','height_concession_pct','staff_concession_ratio']

print('=== OUTCOME COVERAGE AFTER ETL REPAIR ===')
for col in outcome_cols:
    if col not in df.columns:
        print(f'  {col}: MISSING')
        continue
    s = pd.to_numeric(df[col], errors='coerce')
    nz_rows  = (s.abs() > 0).sum()
    nz_cases = df.loc[s.abs() > 0, 'case_number'].nunique()
    mn = s[s > 0].mean() if (s > 0).sum() > 0 else float('nan')
    print(f'  {col:<35} nonzero_rows={nz_rows:>6}  nonzero_cases={nz_cases:>5}  mean_pos={mn:.3f}')

print()
print('=== Delta_Approved_Height detail ===')
dah = pd.to_numeric(df.get('Delta_Approved_Height', pd.Series(dtype=float)), errors='coerce')
print(f'  Non-null rows            : {dah.notna().sum()}')
print(f'  > 0 (approved upzone)    : {(dah > 0).sum()}')
print(f'  < 0 (height reduced)     : {(dah < 0).sum()}')
print(f'  == 0 (no change)         : {(dah == 0).sum()}')
cases_reduced = df.loc[dah < 0, 'case_number'].nunique() if (dah < 0).sum() > 0 else 0
print(f'  Cases with reduction     : {cases_reduced}')
if (dah < 0).sum() > 0:
    print(dah[dah < 0].describe().to_string())

print()
print('=== Staff_Attrition_Height detail ===')
sah = pd.to_numeric(df.get('Staff_Attrition_Height', pd.Series(dtype=float)), errors='coerce')
print(f'  Non-null rows : {sah.notna().sum()}')
print(f'  Nonzero rows  : {(sah.abs() > 0).sum()}')
cn = df.loc[sah.abs() > 0, 'case_number'].nunique() if (sah.abs() > 0).sum() > 0 else 0
print(f'  Cases nonzero : {cn}')
print()

# Best usable outcome: combine net_height_change + Delta_Approved_Height
if 'net_height_change' in df.columns and 'Delta_Approved_Height' in df.columns:
    nhc = pd.to_numeric(df['net_height_change'], errors='coerce').fillna(0)
    dah2 = pd.to_numeric(df['Delta_Approved_Height'], errors='coerce')
    # Use Delta_Approved_Height where available, else net_height_change
    best = dah2.where(dah2.notna(), nhc)
    # Concession = requested height EXCEEDED approved (negative delta = concession)
    # Or use net_height_change > 0 directly (reduction in ft from initial)
    df['height_outcome_best'] = best
    pos = (nhc > 0)
    print('=== RECOMMENDED OUTCOME: net_height_change > 0 ===')
    print(f'  Nonzero rows  : {pos.sum()}')
    print(f'  Cases affected: {df.loc[pos, "case_number"].nunique()}')
    print(f'  Distribution  :')
    print(nhc[nhc > 0].describe().to_string())
