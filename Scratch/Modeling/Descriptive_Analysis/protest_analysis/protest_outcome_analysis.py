import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv', low_memory=False)
enr = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\Zoning_Cases\Processed_Data\CSV\enriched_zoning_data_causal.csv', low_memory=False)

# Merge numeric dimensions onto master
enr2 = enr[['case_number','existing_max_height_ft','proposed_max_height_ft',
             'delta_max_height_ft','existing_max_far','proposed_max_far','delta_max_far']].copy()
df = df.merge(enr2, on='case_number', how='left')

# Protest flag
df['protested'] = df['label_valid_petition_pct'].notna() & (df['label_valid_petition_pct'] > 0)

print(f"Total cases: {len(df):,}")
print(f"Protested:   {df['protested'].sum():,} ({df['protested'].mean()*100:.1f}%)")
print(f"Derived_Status value counts:")
print(df['Derived_Status'].value_counts().head(10))

# ── Ask magnitude: protested vs not ─────────────────────────────────────────
print("\n=== Height ask (delta_max_height_ft) by protest status ===")
print(df.groupby('protested')['delta_max_height_ft'].describe().round(1))

print("\n=== FAR ask (delta_max_far) by protest status ===")
print(df.groupby('protested')['delta_max_far'].describe().round(1))

# ── Final_Zoning vs Requested_Zoning ────────────────────────────────────────
# Check if final differs from requested — proxy for concession
resolved = df[df['Derived_Status'].isin(['Approved','Denied','Withdrawn'])].copy()
resolved['concession'] = resolved['Final_Zoning'] != resolved['Requested_Zoning']
resolved['concession'] = resolved['concession'].astype(float)

print(f"\n=== Concession rate (final != requested) by protest status ===")
print(resolved.groupby('protested')['concession'].agg(['mean','count']).round(3))

# Among approved cases only — did protested ones more often get modified?
approved = resolved[resolved['Derived_Status'] == 'Approved'].copy()
print(f"\n=== Among APPROVED cases only ===")
print(approved.groupby('protested')['concession'].agg(['mean','count']).round(3))

# Denial rate by protest status
print("\n=== Denial rate by protest status ===")
print(resolved.groupby('protested')['Derived_Status'].value_counts(normalize=True).round(3))

# Valid petition pct for protested cases
print("\n=== label_valid_petition_pct distribution (protested cases) ===")
print(df[df['protested']]['label_valid_petition_pct'].describe().round(1))
