import pandas as pd
import numpy as np
import os

ROOT = r"C:\Users\dhl\data\thesis\thesis"
DATA_H0 = os.path.join(ROOT, 'Data', 'Warehouse_As_Of', 'H0_Filing_Master_Enriched.csv')
VOTE_DATA = os.path.join(ROOT, 'Data', 'Zoning_Cases', 'Processed_Data', 'CSV', 'submission_grade_goldmine_tensor.csv')

df = pd.read_csv(DATA_H0, low_memory=False)
print(f"[1] H0 loaded: {len(df)} rows")

df['is_protested'] = df['is_protested'].fillna(0).astype(int)
print(f"[2] Protested in H0: {df['is_protested'].sum()}")

votes = pd.read_csv(VOTE_DATA, usecols=['CASE_NUMBER', 'vote_yes', 'vote_no'])
print(f"[3] Raw vote rows: {len(votes)}, unique cases: {votes['CASE_NUMBER'].nunique()}")

# Apply the fix
votes = votes.groupby('CASE_NUMBER', as_index=False).agg({'vote_yes': 'sum', 'vote_no': 'sum'})
print(f"[4] Deduplicated vote rows: {len(votes)}")

df_left = df.merge(votes, left_on='case_number', right_on='CASE_NUMBER', how='left')
print(f"[5] After LEFT merge: {len(df_left)} rows (should equal {len(df)})")

df_left['is_protested'] = df_left['is_protested'].fillna(0).astype(int)
df_left['is_withdrawn'] = df_left['vote_yes'].isna().astype(int)

df_opposed = df_left[df_left['is_protested'] == 1].copy()
print(f"[6] Opposed cases: {len(df_opposed)} (should equal {df['is_protested'].sum()})")

X_raw = df_opposed.select_dtypes(include=[np.number]).fillna(0)
drop_cols = ['is_withdrawn', 'is_protested', 'case_number', 'organized_opposition', 'TCAD ID', 
             'vote_yes', 'vote_no', 'CASE_NUMBER', 'council_approval', 'ordinance_number', 'council_district']
leak_cols = [c for c in X_raw.columns if c.startswith('tfidf_') or c.startswith('speech_')]
if len(leak_cols) > 0:
    X_raw = X_raw.drop(columns=leak_cols)
X = X_raw.drop(columns=[c for c in drop_cols if c in X_raw.columns], errors='ignore')
print(f"[7] Feature matrix: {X.shape}")
print(f"    Train (90%): {int(len(X) * 0.9)} | Val (10%): {len(X) - int(len(X) * 0.9)}")
