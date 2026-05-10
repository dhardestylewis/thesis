import pandas as pd

df = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv', low_memory=False)
panel = pd.read_csv(r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv', low_memory=False)

# Panel protest cases are the authoritative source
protested_cases = set(panel[panel['petition_event'] == 1]['case_number'].unique())
df['protested'] = df['case_number'].isin(protested_cases)

print('=== Coverage of Requested_Zoning and Final_Zoning ===')
print(f"Requested_Zoning non-null: {df['Requested_Zoning'].notna().sum():,} ({df['Requested_Zoning'].notna().mean()*100:.0f}%)")
print(f"Final_Zoning non-null:     {df['Final_Zoning'].notna().sum():,} ({df['Final_Zoning'].notna().mean()*100:.0f}%)")
print(f"Both non-null:             {(df['Requested_Zoning'].notna() & df['Final_Zoning'].notna()).sum():,}")

# Cases with both populated
both = df[df['Requested_Zoning'].notna() & df['Final_Zoning'].notna()].copy()
print(f"\n=== Of cases with both fields populated: {len(both):,} cases ===")
print(f"Protested: {both['protested'].sum()} ({both['protested'].mean()*100:.1f}%)")

# Did final differ from requested?
both['zoning_changed'] = both['Requested_Zoning'].str.strip() != both['Final_Zoning'].str.strip()
print(f"\n=== Zoning change rate (Requested != Final) ===")
print(both.groupby('protested')['zoning_changed'].agg(['sum','mean','count']).round(3))

# Sample of changed cases
changed = both[both['zoning_changed']][['case_number','protested','Requested_Zoning','Final_Zoning']].head(15)
print(f"\n=== Sample changed cases ===")
print(changed.to_string())

# Among protested cases: what changed?
print(f"\n=== Protested cases with zoning changes ===")
prot_changed = both[both['protested'] & both['zoning_changed']][['case_number','Requested_Zoning','Final_Zoning']]
print(prot_changed.to_string() if len(prot_changed) > 0 else "None found")

# How many unique final zoning values?
print(f"\n=== Top Final_Zoning values for protested cases ===")
print(both[both['protested']]['Final_Zoning'].value_counts().head(10))
print(f"\n=== Top Final_Zoning values for non-protested cases ===")
print(both[~both['protested']]['Final_Zoning'].value_counts().head(10))
