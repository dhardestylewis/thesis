import pandas as pd

df = pd.read_csv(r'C:\Users\dhl\data\Thesis\thesis\Data\final\model_ready_zoning_data.csv', low_memory=False)

print('=== 1. label_valid_petition_pct full distribution ===')
print(df['label_valid_petition_pct'].value_counts(dropna=False).head(10))
print('Unique non-null values:', df['label_valid_petition_pct'].dropna().unique())

print('\n=== 2. Derived_Status full distribution ===')
print(df['Derived_Status'].value_counts(dropna=False))

print('\n=== 3. Sample protested cases ===')
prot = df[df['label_valid_petition_pct'].notna()][['case_number','label_valid_petition_pct','Derived_Status']].head(10)
print(prot.to_string())

print('\n=== 4. Compare panel vs model_ready protest sources ===')
panel = pd.read_csv(
    r'C:\Users\dhl\.gemini\antigravity\brain\1c4648c0-f36a-4614-a8f1-c9e2e5621756\biweekly_panel.csv',
    low_memory=False
)
pet_cases = set(panel[panel['petition_event'] == 1]['case_number'].unique())
model_prot = set(df[df['label_valid_petition_pct'].notna()]['case_number'])
print(f'Panel protest cases:       {len(pet_cases)}')
print(f'model_ready protest cases: {len(model_prot)}')
print(f'Overlap:                   {len(pet_cases & model_prot)}')
print(f'In panel ONLY:             {len(pet_cases - model_prot)}')
print(f'In model_ready ONLY:       {len(model_prot - pet_cases)}')

print('\n=== 5. Panel label_petition_total_pct distribution ===')
pet_rows = panel[panel['petition_event'] == 1][['case_number','label_petition_total_pct','cumulative_petition_pct']]
print(pet_rows['label_petition_total_pct'].describe().round(1))
print('Unique label_petition_total_pct values:', sorted(pet_rows['label_petition_total_pct'].dropna().unique())[:20])
