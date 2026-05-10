import pandas as pd

# Check H0 standardized_tcad_id format variation
enr = pd.read_csv('Data/Warehouse_As_Of/canonical/canonical/H0_Filing_Master_Enriched_v2.csv', low_memory=False)
tcad = enr['standardized_tcad_id'].dropna().astype(str).str.strip()

has_dash = tcad.str.contains('-').sum()
lengths  = tcad.str.len().value_counts().sort_index()
print('=== H0 standardized_tcad_id ===')
print(f'Has dash: {has_dash}, No dash: {(~tcad.str.contains("-")).sum()}')
print('Length distribution:')
print(lengths.to_string())
dash_samples = tcad[tcad.str.contains('-')].head(6).tolist()
nodash_samples = tcad[~tcad.str.contains('-')].head(6).tolist()
print(f'Samples WITH dash:    {dash_samples}')
print(f'Samples WITHOUT dash: {nodash_samples}')

print()
# Check EARS account_number
ears = pd.read_csv('Data/Panel/Intermediate/ears_2022_clean.csv', low_memory=False,
                   usecols=['account_number'])
acct = ears['account_number'].dropna().astype(str).str.strip()
has_dash_ears = acct.str.contains('-').sum()
lengths_ears  = acct.str.len().value_counts().sort_index()
print('=== EARS account_number ===')
print(f'Has dash: {has_dash_ears}, No dash: {(~acct.str.contains("-")).sum()}')
print('Length distribution:')
print(lengths_ears.to_string())
print(f'Sample no-dash: {acct[~acct.str.contains("-")].head(4).tolist()}')
print(f'Sample w/ dash: {acct[acct.str.contains("-")].head(4).tolist()}')

print()
# Check LDB_2016 PID_10
import csv
with open('Data/CoA_Open_Data/LDB_2016_4nsn-uea6.csv', encoding='latin-1') as f:
    reader = csv.DictReader(f)
    pids = []
    for i, row in enumerate(reader):
        if i > 100000: break
        pids.append(str(row.get('PID_10', '')).strip())
pid_series = pd.Series(pids)
print('=== LDB_2016 PID_10 ===')
print('Length distribution:')
print(pid_series.str.len().value_counts().sort_index().to_string())
print(f'Samples: {pid_series[pid_series.str.len() > 0].head(5).tolist()}')
has_dash_ldb = pid_series.str.contains('-').sum()
print(f'Has dash: {has_dash_ldb}')
