import pandas as pd

# Diagnose why EARS is only 17.5% matching
df = pd.read_csv('Data/Warehouse_As_Of/canonical/canonical/H0_Filing_Master_Enriched_v2.csv', low_memory=False)
df = df[df['year'].notna()].copy()

# Sample of TCAD IDs in H0
print('=== H0 standardized_tcad_id samples ===')
tcad_sample = df['standardized_tcad_id'].dropna().head(20).tolist()
print(tcad_sample)

# Sample from EARS
ears = pd.read_csv('Data/Panel/Intermediate/ears_2022_clean.csv', low_memory=False,
                   usecols=['account_number', '_parcel_id_10', 'ears_year'])
print()
print('=== EARS account_number samples ===')
print(ears['account_number'].head(20).tolist())
print()
print('=== EARS account_number stripped ===')
ears_stripped = ears['account_number'].astype(str).str.strip().str.lstrip('0')
print(ears_stripped.head(20).tolist())

# Try to find a matching ID
sample_tcad = str(df['standardized_tcad_id'].dropna().iloc[5])
print(f'\nLooking for TCAD ID: {sample_tcad!r}')
lkp_stripped = sample_tcad.lstrip('0')
matches = ears[ears_stripped == lkp_stripped]
print(f'Matches with lstrip(0): {len(matches)}')

# Try without lstrip
matches2 = ears[ears['account_number'].astype(str).str.strip() == sample_tcad]
print(f'Matches exact: {len(matches2)}')

# Try padded
padded = sample_tcad.zfill(10)
matches3 = ears[ears['account_number'].astype(str).str.strip() == padded]
print(f'Matches zero-padded: {len(matches3)}')

# What format is in EARS exactly?
print()
print('EARS account_number exact format:')
print(repr(ears['account_number'].iloc[100]))
print()

# Check xwalk
xw = pd.read_csv('Data/Panel/Reference/id_crosswalk.csv', low_memory=False)
print('id_crosswalk: parcel_id_10 -> ears_account_number')
print(xw.head(5).to_string())
print()
# Does H0 TCAD ID match crosswalk ears_account_number?
xw['_norm'] = xw['ears_account_number'].astype(str).str.strip().str.lstrip('0')
sample_norm = sample_tcad.lstrip('0')
match_xw = xw[xw['_norm'] == sample_norm]
print(f'TCAD {sample_tcad!r} in crosswalk: {len(match_xw)}')
if len(match_xw) > 0:
    print(match_xw[['parcel_id_10','ears_account_number']].head(3).to_string())
