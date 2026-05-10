import pandas as pd

panel = pd.read_csv(r'c:\Users\dhl\data\Thesis\thesis\Scratch\Modeling\Causal_Inference\05_G_Computation_LSTMs\biweekly_panel.csv', low_memory=False)

print('=== 1. PDF Height Features ===')
pdf_cols = [c for c in panel.columns if c.startswith('pdf_') and 'zoning' not in c]
for c in pdf_cols:
    nn = panel[c].notna().sum()
    if nn > 0:
        desc = panel[c].describe()
        print(f'{c} ({nn:,} non-null): Mean {desc["mean"]:.1f}, Min {desc["min"]:.0f}, Max {desc["max"]:.0f}')

print('\n=== 2. Cumulative NLP Features ===')
nlp_cols = ['nlp_document_count', 'nlp_oppose_hits', 'council_nlp_document_count', 'council_nlp_oppose_hits']
for c in nlp_cols:
    if c in panel.columns:
        nn = (panel[c] > 0).sum()
        if nn > 0:
            desc = panel[panel[c] > 0][c].describe()
            print(f'{c} (Rows > 0: {nn:,}): Mean {desc["mean"]:.1f}, Max {desc["max"]:.0f}')

print('\n=== 3. Petition Features ===')
pet_cols = ['petition_pct_this_period', 'cumulative_petition_pct']
for c in pet_cols:
    if c in panel.columns:
        nn_gt0 = (panel[c].fillna(0) > 0).sum()
        if nn_gt0 > 0:
            desc = panel[panel[c] > 0][c].describe()
            print(f'{c} (Rows > 0: {nn_gt0:,}): Mean {desc["mean"]:.2f}, Max {desc["max"]:.1f}')
        else:
            print(f'{c}: 0 non-zero rows')

print('\n=== 4. Appraisal / ACS (Sanity check) ===')
for c in ['appraised_value', 'median_household_income']:
    if c in panel.columns:
        desc = panel[c].dropna().describe()
        print(f'{c}: Mean {desc["mean"]/1000:.1f}k, Min {desc["min"]}, Max {desc["max"]}')
