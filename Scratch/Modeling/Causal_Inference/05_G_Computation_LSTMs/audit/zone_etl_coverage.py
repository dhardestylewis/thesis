"""
zone_etl_coverage.py
Audit zoning ETL coverage and classify commission transcript corpus by document type.
"""
import pandas as pd
import numpy as np
import re

mrd   = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv', low_memory=False)
comm  = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Data\interim\commission_transcripts.csv', low_memory=False)

print('=== ZONE ETL COVERAGE ===')
print(f'Total cases in model_ready_zoning_data: {len(mrd)}')
print()

rz_pop = (mrd['Requested_Zoning'].fillna('').str.strip() != '').sum()
fz_pop = (mrd['Final_Zoning'].fillna('').str.strip() != '').sum()
iz_pop = (mrd['Initial_Zoning'].fillna('').str.strip() != '').sum()
print(f'Requested_Zoning populated : {rz_pop:>5} / {len(mrd)} ({rz_pop/len(mrd):.1%})')
print(f'Initial_Zoning populated   : {iz_pop:>5} / {len(mrd)} ({iz_pop/len(mrd):.1%})')
print(f'Final_Zoning populated     : {fz_pop:>5} / {len(mrd)} ({fz_pop/len(mrd):.1%})')
print()

approved = mrd[mrd['detailed_status'] == 'Approved']
fz_appr  = (approved['Final_Zoning'].fillna('').str.strip() != '').sum()
print(f'Approved cases              : {len(approved)}')
print(f'Final_Zoning among approved : {fz_appr} / {len(approved)} ({fz_appr/len(approved):.1%})')
print()

print('Height lookup coverage (from zoning code -> LDC table):')
for col in ['Requested_max_height_ft','Initial_max_height_ft',
            'Approved_max_height_ft','Staff_max_height_ft']:
    s = pd.to_numeric(mrd[col], errors='coerce')
    print(f'  {col:<35} {s.notna().sum():>5} ({s.notna().mean():.1%})')

print()
print('=== COMMISSION TRANSCRIPT CORPUS ===')
print(f'Total docs in commission_transcripts.csv: {len(comm)}')

def classify_doc(fn):
    fn = str(fn).lower()
    if 'backup' in fn:       return 'backup_staff_report'
    if 'staff report' in fn or 'staff_report' in fn: return 'staff_report'
    if 'ordinance' in fn:    return 'ordinance'
    if 'exhibit' in fn:      return 'exhibit'
    if 'presentation' in fn: return 'presentation'
    if 'minute' in fn:       return 'minutes'
    if 'agenda' in fn:       return 'agenda'
    return 'other'

fnames = comm['Filename'].fillna('').astype(str)
doc_types = fnames.apply(classify_doc)
print()
print('Document type breakdown:')
vc = doc_types.value_counts()
for dtype, count in vc.items():
    pct = count / len(comm)
    print(f'  {dtype:<25} {count:>5} ({pct:.1%})')

print()
backup_docs = comm[doc_types == 'backup_staff_report']
print(f'Backup/staff report docs: {len(backup_docs)}')
print()
print('Sample backup filenames:')
for fn in backup_docs['Filename'].dropna().head(20):
    print(f'  {fn}')

print()
print('=== HOW BACKUP DOCS ARE LINKED ===')
# Check if planning_commission_index or zoning_platting_commission_index have doc type info
import os
for idx_name in ['planning_commission_index.csv', 'zoning_platting_commission_index.csv']:
    path = os.path.join(r'c:\Users\dhl\data\Thesis\thesis\Data\raw\indices', idx_name)
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        print(f'{idx_name}: {df.shape}')
        print(f'  Columns: {list(df.columns)}')
        if 'Doc_Text' in df.columns:
            def classify_doc_text(t):
                t = str(t).lower()
                if 'backup' in t:       return 'backup'
                if 'agenda' in t:       return 'agenda'
                if 'minute' in t:       return 'minutes'
                if 'ordinance' in t:    return 'ordinance'
                return 'other'
            dt_types = df['Doc_Text'].apply(classify_doc_text).value_counts()
            print(f'  Doc_Text types:')
            for k, v in dt_types.items():
                print(f'    {k:<20} {v:>5}')
        print()
    else:
        print(f'{idx_name}: NOT FOUND at {path}')
        # Search for it
        for root, dirs, files in os.walk(r'c:\Users\dhl\data\Thesis\thesis\Data'):
            for f in files:
                if idx_name in f:
                    print(f'  Found at: {os.path.join(root, f)}')
